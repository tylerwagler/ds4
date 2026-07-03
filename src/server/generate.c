#include "ds4_server_internal.h"



static bool thinking_tail_ends_with(const thinking_state *st, const char *s) {
    int n = (int)strlen(s);
    return st->tail_len >= n && !memcmp(st->tail + st->tail_len - n, s, (size_t)n);
}



void thinking_state_feed(thinking_state *st, const char *p, size_t len) {
    if (!st || !p) return;
    for (size_t i = 0; i < len; i++) {
        if (st->tail_len == (int)sizeof(st->tail)) {
            memmove(st->tail, st->tail + 1, sizeof(st->tail) - 1);
            st->tail_len--;
        }
        st->tail[st->tail_len++] = p[i];
        if (thinking_tail_ends_with(st, "<think>")) st->inside = true;
        else if (thinking_tail_ends_with(st, "</think>")) st->inside = false;
    }
}



thinking_state thinking_state_from_prompt(const request *r) {
    thinking_state st = {0};
    if (r && r->prompt_text) {
        thinking_state_feed(&st, r->prompt_text, strlen(r->prompt_text));
    } else if (r && ds4_think_mode_enabled(r->think_mode)) {
        st.inside = true;
    }
    return st;
}



/* Live recovery for a tool call started inside an unclosed <think> block.
 *
 * The model sometimes opens a DSML stanza without closing its thinking first.
 * Waiting for a </think> that never comes stalls the turn: the marker is never
 * scanned as executable and the block is dropped at parse time.  Instead of
 * rewriting sampled context, recover forward: force-feed "</think>" plus a
 * blank line and let the model continue.  Measured on the real model, that
 * position predicts a fresh stanza opening so strongly that the model
 * restarts the call cleanly on the executable side of the close.  Re-emitting
 * the stanza opening ourselves was tried and is counterproductive: with the
 * dangling opening right before the close and a forced copy right after it,
 * the model reads the call as already made and ends the turn.  The dangling
 * opening stays harmlessly inside reasoning.
 *
 * Detection works on accumulated text, so the tokenization of the marker does
 * not matter, and it triggers only on a complete stanza opening: a lone "<"
 * or a partial marker keeps decoding untouched, while *scan_from holds back
 * far enough that an opening split across future tokens is still seen from
 * its first byte.  The forced text is tokenized with the rendered-chat
 * tokenizer so </think> maps to its special token.
 *
 * Returns 1 when an injection was performed (text extended, thinking closed),
 * 0 when there is nothing to do or no budget, -1 on eval failure. */
static int chat_think_tool_recovery(server *s,
                                    buf *text,
                                    thinking_state *thinking,
                                    size_t *scan_from,
                                    int *completion,
                                    int max_tokens,
                                    char *err,
                                    size_t errlen) {
    if (!thinking->inside || !text->ptr) return 0;
    if (*scan_from > text->len) *scan_from = text->len;
    if (!find_any_tool_start(text->ptr + *scan_from)) {
        const size_t hold = 80; /* > longest stanza opening */
        *scan_from = text->len > hold ? text->len - hold : 0;
        return 0;
    }

    const char *inject = "</think>\n\n";
    const size_t inject_len = strlen(inject);
    ds4_tokens toks = {0};
    ds4_tokenize_rendered_chat(s->engine, inject, &toks);

    const int room = ds4_session_ctx(s->session) - ds4_session_pos(s->session);
    if (toks.len <= 0 ||
        toks.len >= room ||
        *completion + toks.len >= max_tokens) {
        /* Not enough budget to recover; leave the stream as generated and let
         * the parse-time fallback deal with it.  Skip past this marker so the
         * scan does not retry it every token. */
        ds4_tokens_free(&toks);
        *scan_from = text->len;
        return 0;
    }

    for (int i = 0; i < toks.len; i++) {
        if (ds4_session_eval(s->session, toks.v[i], err, errlen) != 0) {
            ds4_tokens_free(&toks);
            return -1;
        }
        (*completion)++;
    }
    buf_append(text, inject, inject_len);
    thinking_state_feed(thinking, inject, inject_len);
    *scan_from = text->len;
    ds4_tokens_free(&toks);
    return 1;
}



static char *rendered_chat_system_region(const char *prompt_text) {
    if (!prompt_text) return xstrdup("");
    const char *p = prompt_text;
    const char *bos = "<｜begin▁of▁sentence｜>";
    const size_t bos_len = strlen(bos);
    if (!strncmp(p, bos, bos_len)) p += bos_len;
    const char *max_prefix = ds4_think_max_prefix();
    const size_t max_prefix_len = strlen(max_prefix);
    if (max_prefix_len && !strncmp(p, max_prefix, max_prefix_len)) {
        p += max_prefix_len;
    }
    while (*p && isspace((unsigned char)*p)) p++;

    const char *user = strstr(p, "<｜User｜>");
    const char *assistant = strstr(p, "<｜Assistant｜>");
    const char *end = NULL;
    if (user && assistant) end = user < assistant ? user : assistant;
    else end = user ? user : assistant;
    if (!end) end = p + strlen(p);
    while (end > p && isspace((unsigned char)end[-1])) end--;
    return xstrndup(p, (size_t)(end - p));
}



char *build_invalid_dsml_tool_error_suffix(const request *r,
                                                  const thinking_state *thinking,
                                                  const char *detail) {
    char *system = rendered_chat_system_region(r ? r->prompt_text : NULL);
    buf tool_error = {0};
    buf_puts(&tool_error, "Tool error: invalid DSML tool call");
    if (detail && detail[0]) {
        buf_puts(&tool_error, ": ");
        buf_puts(&tool_error, detail);
    }
    buf_puts(&tool_error,
             "\nThe previous assistant output was not executed because the DSML syntax was malformed. "
             "Emit a new valid DSML tool call, or answer normally if no tool is needed.");
    if (system && system[0]) {
        buf_puts(&tool_error, "\n\nSystem prompt reminder:\n");
        buf_puts(&tool_error, system);
    }

    buf suffix = {0};
    if (r && ds4_think_mode_enabled(r->think_mode) && thinking && thinking->inside) {
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, "<｜end▁of▁sentence｜><｜User｜><tool_result>");
    append_tool_result_text(&suffix, tool_error.ptr ? tool_error.ptr : "");
    buf_puts(&suffix, "</tool_result><｜Assistant｜>");
    buf_puts(&suffix, r && ds4_think_mode_enabled(r->think_mode) ? "<think>" : "</think>");

    free(system);
    buf_free(&tool_error);
    return buf_take(&suffix);
}



static bool append_rendered_suffix_to_live_session(server *s, const char *suffix,
                                                   int *tokens_appended,
                                                   char *err, size_t errlen) {
    if (tokens_appended) *tokens_appended = 0;
    if (!s || !suffix || !suffix[0]) return true;
    const ds4_tokens *live = ds4_session_tokens(s->session);
    if (!live) {
        if (err && errlen) snprintf(err, errlen, "live session is unavailable");
        return false;
    }

    ds4_tokens target = {0};
    build_prompt_from_exact_prefix_and_text_suffix(s->engine, live, suffix, &target);
    const int before = ds4_session_pos(s->session);
    bool ok = ds4_session_sync(s->session, &target, err, errlen) == 0;
    if (ok && tokens_appended) {
        int delta = ds4_session_pos(s->session) - before;
        *tokens_appended = delta > 0 ? delta : 0;
    }
    ds4_tokens_free(&target);
    return ok;
}



static bool continue_after_invalid_dsml(server *s, const request *r,
                                        const thinking_state *thinking,
                                        const char *detail,
                                        int *tokens_appended,
                                        char *err, size_t errlen) {
    char *suffix = build_invalid_dsml_tool_error_suffix(r, thinking, detail);
    bool ok = append_rendered_suffix_to_live_session(s, suffix,
                                                     tokens_appended,
                                                     err, errlen);
    free(suffix);
    return ok;
}



bool should_remember_thinking_checkpoint(const request *r,
                                                const thinking_state *thinking,
                                                const char *finish) {
    if (!r || r->kind != REQ_CHAT || r->has_tools) return false;
    if (r->prompt_preserves_reasoning) return false;
    if (!ds4_think_mode_enabled(r->think_mode)) return false;
    if (finish && (!strcmp(finish, "error") || !strcmp(finish, "length"))) return false;
    if (thinking && thinking->inside) return false;
    return true;
}



static void log_tool_calls_summary(const char *ctx, const tool_calls *calls,
                                   bool responses_protocol) {
    if (!calls || calls->len == 0) return;
    buf names = {0};
    buf ids = {0};
    for (int i = 0; i < calls->len; i++) {
        if (i) buf_putc(&names, ',');
        if (i) buf_putc(&ids, ',');
        buf_puts(&names, calls->v[i].name ? calls->v[i].name : "?");
        buf_puts(&ids, calls->v[i].id ? calls->v[i].id : "?");
    }
    char flags[32];
    log_flags(flags, sizeof(flags), responses_protocol, false, false, false, false);
    server_log(DS4_LOG_TOOL,
               "ds4-server: tool calls ctx=%s%s%s n=%d raw_dsml=%d ids=[%s] names=[%s]",
               ctx,
               flags[0] ? " " : "",
               flags,
               calls->len,
               calls->raw_dsml && calls->raw_dsml[0] ? 1 : 0,
               ids.ptr ? ids.ptr : "",
               names.ptr ? names.ptr : "");
    buf_free(&ids);
    buf_free(&names);
}



static void server_progress_cb(void *ud, const char *event, int current, int total) {
    server_prefill_progress *p = ud;
    if (!p || !event) return;
    const bool is_chunk = strcmp(event, "prefill_chunk") == 0;
    const bool is_display = strcmp(event, "prefill_display") == 0;
    if (!is_chunk && !is_display) return;

    double now = server_now_sec();
    /* Keep the HTTP/SSE connection alive while prefill runs.  We write the SSE
     * response headers the first time the callback fires and then emit a
     * comment line (`:` prefix, ignored by SSE clients) every few seconds.
     * Best-effort: if the client has already gone away, the writes fail
     * silently and the outer code will discover the closed socket the next
     * time it tries to stream a real event. */
    if (p->stream && p->fd >= 0 && !p->stream_failed) {
        if (!p->headers_sent) {
            p->headers_sent = true;
            if (sse_headers(p->fd, p->enable_cors)) {
                p->last_keepalive = now;
            } else {
                p->stream_failed = true;
            }
        } else if (now - p->last_keepalive >= 5.0) {
            static const char ka[] = ": prefill\n\n";
            if (send_all(p->fd, ka, sizeof(ka) - 1)) {
                p->last_keepalive = now;
            } else {
                p->stream_failed = true;
            }
        }
    }
    if (is_display) return;
    double elapsed = now - p->t0;
    if (p->seen && current == p->last_current) {
        if (p->srv && current > p->cached_tokens) {
            kv_cache_maybe_store_continued(p->srv);
        }
        return;
    }
    int display_start = p->cached_tokens;
    if (display_start < 0 || display_start > p->prompt_tokens) display_start = 0;
    int display_total = p->prompt_tokens - display_start;
    if (display_total <= 0) {
        display_start = 0;
        display_total = p->prompt_tokens > total ? p->prompt_tokens : total;
    }
    int display_current = current - display_start;
    if (display_current < 0) display_current = 0;
    if (display_current > display_total) display_current = display_total;
    double pct = display_total > 0 ? 100.0 * (double)display_current / (double)display_total : 100.0;
    double avg_tps = elapsed > 0.0 ? (double)display_current / elapsed : 0.0;
    int interval_tokens = p->seen ? current - p->last_current : 0;
    if (interval_tokens < 0) interval_tokens = 0;
    double interval_s = p->seen ? now - p->last_t : 0.0;
    double chunk_tps = interval_s > 0.0 ? (double)interval_tokens / interval_s : 0.0;
    p->last_current = current;
    p->last_t = now;
    p->seen = true;
    char flags[64];
    log_flags(flags, sizeof(flags), p->responses_protocol,
              p->has_tools, false, false, false);
    const char *phase = p->phase ? p->phase : "prefill";
    server_log(DS4_LOG_PREFILL,
               "ds4-server: %s ctx=%s%s%s %s chunk %d/%d (%.1f%%) chunk=%.2f t/s avg=%.2f t/s %.3fs",
               p->kind == REQ_CHAT ? "chat" : "completion",
               p->ctx,
               flags[0] ? " " : "",
               flags,
               phase,
               display_current,
               display_total,
               pct,
               chunk_tps,
               avg_tps,
               elapsed);
    if (p->srv && current > p->cached_tokens) {
        kv_cache_maybe_store_continued(p->srv);
    }
}



static void send_prefill_failure_response(server *s, const job *j,
                                          const server_prefill_progress *progress,
                                          const char *ctx, const char *flags,
                                          const char *err) {
    const char *kind = j->req.kind == REQ_CHAT ? "chat" : "completion";
    if (j->req.stream && progress && progress->headers_sent) {
        if (progress->stream_failed) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s%s%s prefill failed after stream closed: %s",
                       kind, ctx, flags && flags[0] ? " " : "",
                       flags && flags[0] ? flags : "", err);
            return;
        }
        if (!sse_error_event(j->fd, &j->req, err)) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s%s%s prefill SSE error failed: %s",
                       kind, ctx, flags && flags[0] ? " " : "",
                       flags && flags[0] ? flags : "", err);
        }
        return;
    }
    http_error(j->fd, s->enable_cors, 500, err);
}



char *build_tool_checkpoint_suffix(const request *r, const char *content,
                                          const char *reasoning, const tool_calls *calls) {
    buf suffix = {0};
    if (ds4_think_mode_enabled(r->think_mode)) {
        buf_puts(&suffix, reasoning ? reasoning : "");
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, content ? content : "");
    append_dsml_tool_calls_text(&suffix, calls);
    buf_puts(&suffix, "<｜end▁of▁sentence｜>");
    return buf_take(&suffix);
}



char *build_responses_visible_assistant_suffix(const request *r,
                                                      const char *content,
                                                      const char *reasoning,
                                                      const tool_calls *calls) {
    buf suffix = {0};
    /* This suffix mirrors what a Responses client can replay, not necessarily
     * every token in KV.  Hidden reasoning stays live in the session unless the
     * next client replay is expected to include it.  In practice, pi replays
     * reasoning summaries for tool-call turns, but not for final assistant
     * answers; Codex currently requests no summaries at all.  So only include
     * reasoning in the remembered visible prefix when this assistant turn ended
     * in tool calls.  A client that does replay final-answer reasoning will not
     * match this visible shortcut and can still use exact token-prefix replay. */
    if (ds4_think_mode_enabled(r->think_mode)) {
        if (r->reasoning_summary_emit && calls && calls->len > 0) {
            buf_puts(&suffix, reasoning ? reasoning : "");
        }
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, content ? content : "");
    append_dsml_tool_calls_text(&suffix, calls);
    buf_puts(&suffix, "<｜end▁of▁sentence｜>");
    return buf_take(&suffix);
}



/* In thinking mode without tools, old assistant reasoning is intentionally not
 * rendered back into later prompts.  The sampled live graph still contains the
 * reasoning bytes, so the next request would miss the session cache even though
 * the visible conversation prefix is logically the same.
 *
 *   prompt-without-final-<think> + </think> + visible-content + eos
 *
 * is exactly the visible prefix that render_chat_prompt_text() will produce on
 * the next turn.  Do not rebuild the KV cache to erase hidden reasoning here:
 * that caused long post-answer pauses and threw away useful sampled state.
 * Instead, remember the visible bytes as a key for the current sampled frontier.
 * The next request can then continue from live KV while tokenizing only the new
 * visible suffix. */
char *build_toolless_thinking_visible_text(const request *r,
                                                  const char *content) {
    if (!r || !r->prompt_text) return NULL;
    if (!ds4_think_mode_enabled(r->think_mode)) return NULL;

    size_t pt_len = strlen(r->prompt_text);
    const char *think_tag = "<think>";
    size_t tag_len = strlen(think_tag);
    if (pt_len < tag_len ||
        memcmp(r->prompt_text + pt_len - tag_len, think_tag, tag_len) != 0) {
        return NULL;
    }

    buf visible = {0};
    buf_append(&visible, r->prompt_text, pt_len - tag_len);
    buf_puts(&visible, "</think>");
    buf_puts(&visible, content ? content : "");
    buf_puts(&visible, "<｜end▁of▁sentence｜>");
    return buf_take(&visible);
}



static void remember_thinking_checkpoint(server *s, const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content) {
    char *visible = build_toolless_thinking_visible_text(&j->req, content);
    if (!visible) return;

    thinking_live_remember(s, visible);
    server_log(DS4_LOG_KVCACHE,
               "ds4-server: thinking live checkpoint remembered ctx=%s live=%d visible=%zu",
               ctx, ds4_session_pos(s->session), strlen(visible));
    trace_event(s, trace_id,
                "thinking live checkpoint remembered: live=%d visible=%zu",
                ds4_session_pos(s->session), strlen(visible));
    free(visible);
}



/* After a successful tool-call finish, make the live checkpoint match what the
 * next request will render.  Usually that is just the exact DSML remembered by
 * tool id.  If a client sends a tool call without an id we know, the fallback
 * renderer still builds valid DSML from JSON, and this function either rewrites
 * the short suffix in place or reloads an older disk checkpoint before replay. */
static void canonicalize_tool_checkpoint(server *s, const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content,
                                         const char *reasoning, const tool_calls *calls) {
    if (!calls || calls->len == 0 || !j->req.prompt_text) return;

    char *suffix_text = build_tool_checkpoint_suffix(&j->req, content, reasoning, calls);

    buf rendered = {0};
    buf_puts(&rendered, j->req.prompt_text);
    buf_puts(&rendered, suffix_text);

    ds4_tokens canonical = {0};
    ds4_tokenize_rendered_chat(s->engine, rendered.ptr ? rendered.ptr : "", &canonical);
    const int live_len = ds4_session_pos(s->session);
    const int common = ds4_session_common_prefix(s->session, &canonical);
    if (common == live_len && canonical.len == live_len) goto done;

    size_t live_text_len = 0;
    char *live_text = render_tokens_text(s->engine, ds4_session_tokens(s->session), &live_text_len);
    if (live_text_len == rendered.len &&
        (live_text_len == 0 || memcmp(live_text, rendered.ptr, live_text_len) == 0))
    {
        /* The graph already represents the bytes the next request will render.
         * Token-level canonicalization would only replace a valid sampled
         * history with a different BPE spelling of the same transcript. */
        free(live_text);
        goto done;
    }
    free(live_text);

    if (common < j->req.prompt.len) {
        trace_event(s, trace_id,
                    "tool checkpoint canonicalization skipped: common=%d prompt=%d live=%d canonical=%d",
                    common, j->req.prompt.len, live_len, canonical.len);
        goto done;
    }

    char err[160] = {0};
    ds4_session_rewrite_result rr =
        ds4_session_rewrite_from_common(s->session, &canonical, common,
                                        err, sizeof(err));
    if (rr == DS4_SESSION_REWRITE_OK) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d",
                   ctx, common, live_len, canonical.len);
        trace_event(s, trace_id,
                    "tool checkpoint canonicalized: common=%d live=%d canonical=%d",
                    common, live_len, canonical.len);
    } else if (rr == DS4_SESSION_REWRITE_REBUILD_NEEDED) {
        /* The generated DSML suffix and the canonical prompt share a prefix,
         * but the generated tail is too large to overwrite safely inside the
         * live raw-window ring.  Prefer an older disk checkpoint over replaying
         * a very long conversation from token zero. */
        char *path = NULL;
        ds4_tokens effective = {0};
        int loaded = kv_cache_try_load_text(s, rendered.ptr ? rendered.ptr : "",
                                            &effective, &path, NULL, false);
        if (loaded == 0) ds4_session_invalidate(s->session);

        char sync_err[160] = {0};
        const ds4_tokens *sync_prompt = loaded > 0 ? &effective : &canonical;
        char rebuild_ctx[48];
        request_ctx_span(rebuild_ctx, sizeof(rebuild_ctx), loaded, sync_prompt->len);
        int replay_tokens = sync_prompt->len - loaded;
        if (replay_tokens < 0) replay_tokens = sync_prompt->len;
        int canonical_tail_tokens = canonical.len - common;
        if (canonical_tail_tokens < 0) canonical_tail_tokens = canonical.len;
        int discarded_live_tokens = live_len - common;
        if (discarded_live_tokens < 0) discarded_live_tokens = 0;
        const char *source = loaded > 0 ? "disk" : "full";
        const double rebuild_t0 = server_now_sec();
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalization needs %d tokens rebuild ctx=%s request_ctx=%s reason=canonical-tail-rewrite tail=%d discard=%d common=%d live=%d target=%d cached=%d source=%s%s%s",
                   replay_tokens,
                   rebuild_ctx,
                   ctx,
                   canonical_tail_tokens,
                   discarded_live_tokens,
                   common,
                   live_len,
                   canonical.len,
                   loaded,
                   source,
                   path ? " file=" : "",
                   path ? path : "");
        server_prefill_progress rebuild_progress = {
            .srv = s,
            .kind = j->req.kind,
            .prompt_tokens = sync_prompt->len,
            .cached_tokens = loaded,
            .phase = "tool checkpoint rebuild",
            .has_tools = j->req.has_tools,
            .t0 = rebuild_t0,
            .fd = j->fd,
            .stream = j->req.stream,
            .enable_cors = s->enable_cors,
            /* Tool checkpoint rebuild only runs after the response stream is
             * already in flight, so the SSE headers were sent long ago.
             * Pre-arm the flag so the progress callback only emits keepalive
             * comments and never tries to write a second set of headers. */
            .headers_sent = true,
        };
        snprintf(rebuild_progress.ctx, sizeof(rebuild_progress.ctx), "%s", rebuild_ctx);
        ds4_session_set_progress(s->session, server_progress_cb, &rebuild_progress);
        ds4_session_set_display_progress(s->session, server_progress_cb, &rebuild_progress);
        if (ds4_session_sync(s->session, sync_prompt, sync_err, sizeof(sync_err)) == 0) {
            ds4_session_set_progress(s->session, NULL, NULL);
            ds4_session_set_display_progress(s->session, NULL, NULL);
            const double rebuild_sec = server_now_sec() - rebuild_t0;
            if (loaded > 0) {
                server_log(DS4_LOG_KVCACHE,
                           "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=disk cached=%d replay=%d target=%d %.3fs",
                           rebuild_ctx, ctx, loaded, replay_tokens, canonical.len, rebuild_sec);
                trace_event(s, trace_id,
                            "tool checkpoint canonicalized via disk: common=%d live=%d canonical=%d cached=%d file=%s",
                            common, live_len, canonical.len, loaded, path ? path : "");
            } else {
                server_log(DS4_LOG_KVCACHE,
                           "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=full cached=0 replay=%d target=%d %.3fs",
                           rebuild_ctx, ctx, replay_tokens, canonical.len, rebuild_sec);
                trace_event(s, trace_id,
                            "tool checkpoint canonicalized via rebuild: common=%d live=%d canonical=%d reason=%s",
                            common, live_len, canonical.len, err);
            }
        } else {
            ds4_session_set_progress(s->session, NULL, NULL);
            ds4_session_set_display_progress(s->session, NULL, NULL);
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: tool checkpoint rebuild failed ctx=%s request_ctx=%s source=%s cached=%d replay=%d target=%d error=\"%s\"",
                       rebuild_ctx, ctx, source, loaded, replay_tokens,
                       canonical.len, sync_err);
            trace_event(s, trace_id, "tool checkpoint canonicalization failed after rebuild request: %s", sync_err);
        }
        ds4_tokens_free(&effective);
        free(path);
    } else {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalization failed ctx=%s common=%d live=%d canonical=%d error=\"%s\"",
                   ctx, common, live_len, canonical.len, err);
        trace_event(s, trace_id, "tool checkpoint canonicalization failed: %s", err);
    }

done:
    ds4_tokens_free(&canonical);
    buf_free(&rendered);
    free(suffix_text);
}



bool should_canonicalize_tool_checkpoint(const server *s, const tool_calls *calls) {
    if (!calls || calls->len == 0) return false;
    if (s && !s->disable_exact_dsml_tool_replay &&
        calls->raw_dsml && calls->raw_dsml[0])
    {
        return false;
    }
    return true;
}



/* Execute one request on the worker-owned session.
 *
 * Clients resend full prompts as text.  The worker first tries the old exact
 * token-prefix hit, then a rendered-text prefix hit for the live checkpoint,
 * then disk text-prefix restart snapshots, then a cold prefill.  On text-prefix
 * hits we build a fresh effective prompt from the checkpoint's exact token
 * history plus a newly tokenized string suffix; the canonical full-prompt
 * tokens are not sliced because BPE may merge across the byte boundary.  Cold
 * prompt caching is handled before generation: if the stable checkpoint is
 * shorter than the full prompt, we prefill to that boundary, store it, and
 * immediately continue to the real prompt.  The live graph therefore always
 * moves forward. */
void generate_job(server *s, job *j) {
    char err[160];
    err[0] = '\0';
    const int old_pos = ds4_session_pos(s->session);
    const int common = ds4_session_common_prefix(s->session, &j->req.prompt);
    trace_cache_diag cache_diag = {0};
    trace_cache_capture(&cache_diag, ds4_session_tokens(s->session),
                        &j->req.prompt, old_pos, common);
    ds4_tokens effective_prompt = {0};
    const ds4_tokens *prompt_for_sync = &j->req.prompt;
    const bool responses_protocol = j->req.api == API_RESPONSES;
    bool responses_live_continuation = false;
    bool anthropic_live_continuation = false;
    bool thinking_live_continuation = false;
    const char *responses_live_match = NULL;
    int responses_live_match_ids = 0;
    int anthropic_live_match_ids = 0;
    /* Responses gets the first chance to continue from live state.  This is
     * the whole point of the API shape: a request that is bound to prior live
     * output by visible transcript or tool call ids does not need to prove an
     * exact token-prefix match.  Exact token/text/disk matching remains the
     * fallback when the live state is absent or no longer describes the
     * request. */
    int cached = responses_live_visible_prefix_prompt(s, &j->req, old_pos,
                                                      &effective_prompt);
    const char *cache_source = cached > 0 ? "responses-visible" : "none";
    if (cached > 0) {
        responses_live_match = "visible-prefix";
        if (responses_live_matches_request(s, &j->req.responses_live_call_ids,
                                           old_pos))
        {
            responses_live_match_ids = j->req.responses_live_call_ids.len;
        }
    }
    if (cached == 0) {
        cached = responses_live_continuation_prompt(s, &j->req, old_pos,
                                                    &effective_prompt,
                                                    &responses_live_match_ids);
        cache_source = cached > 0 ? "responses-tool-output" : "none";
        if (cached > 0) responses_live_match = "tool-output-ids";
    }
    if (cached > 0) {
        responses_live_continuation = true;
        prompt_for_sync = &effective_prompt;
    } else {
        cached = anthropic_live_continuation_prompt(s, &j->req, old_pos,
                                                    &effective_prompt,
                                                    &anthropic_live_match_ids);
        if (cached > 0) {
            anthropic_live_continuation = true;
            cache_source = "anthropic-tool-output";
            prompt_for_sync = &effective_prompt;
        }
    }
    if (cached == 0 && responses_protocol &&
        j->req.responses_requires_live_tool_state)
    {
        /* The parser saw a valid live call_id, but by worker execution time the
         * live frontier no longer matches.  Since the request did not replay
         * the prior assistant call, there is no stateless prefix to match and
         * no disk key to search by. */
        ds4_tokens_free(&effective_prompt);
        http_error(j->fd, s->enable_cors, 409,
                   "Responses continuation state is not available; retry by replaying the full input history");
        return;
    } else if (cached == 0 && j->req.api == API_ANTHROPIC &&
               j->req.anthropic_requires_live_tool_state)
    {
        ds4_tokens_free(&effective_prompt);
        http_error(j->fd, s->enable_cors, 409,
                   "Anthropic continuation state is not available; retry by replaying the full messages history");
        return;
    } else if (cached == 0) {
        cached = common == old_pos && j->req.prompt.len >= old_pos ? common : 0;
        cache_source = cached > 0 ? "memory-token" : "none";
    }
    if (cached == 0) {
        int thinking_cached =
            thinking_live_visible_prefix_prompt(s, &j->req, old_pos,
                                                &effective_prompt);
        if (thinking_cached > 0) {
            cached = thinking_cached;
            cache_source = "thinking-visible";
            thinking_live_continuation = true;
            prompt_for_sync = &effective_prompt;
        }
    }
    int disk_cached = 0;
    char *disk_cache_path = NULL;
    uint8_t disk_cache_ext_flags = 0;
    if (cached == 0) {
        int text_cached = live_text_prefix_prompt(s, &j->req, &effective_prompt);
        if (text_cached > 0) {
            cached = text_cached;
            cache_source = "memory-text";
            prompt_for_sync = &effective_prompt;
        }
    }
    if (cached == 0 && old_pos > 0) {
        server_log(DS4_LOG_WARNING,
                   "ds4-server: live kv cache miss%s live=%d prompt=%d common=%d reason=%s",
                   responses_protocol ? " RESPPROTO" : "",
                   old_pos, j->req.prompt.len, common,
                   trace_cache_miss_reason(&cache_diag));
    }
    if (cached == 0) s->kv.continued_last_store_tokens = 0;
    if (s->kv.enabled && cached == 0 && old_pos >= s->kv.opt.min_tokens) {
        /* Loading a disk snapshot replaces the live Metal session.  Persist the
         * current checkpoint first, otherwise a cache hit for an older prefix
         * would silently discard the newer conversation state. */
        kv_cache_store_current(s, "evict");
    }
    if (cached == 0) {
        disk_cached = kv_cache_try_load(s, &j->req, &effective_prompt,
                                        &disk_cache_path,
                                        &disk_cache_ext_flags);
        if (disk_cached > 0) {
            cached = disk_cached;
            cache_source = "disk-text";
            prompt_for_sync = &effective_prompt;
        }
    }
    const bool responses_reasoning_state_preserved =
        cached > 0 &&
        ((!strcmp(cache_source, "responses-visible") ||
          !strcmp(cache_source, "responses-tool-output")) ||
         (!strcmp(cache_source, "disk-text") &&
          (disk_cache_ext_flags & KV_EXT_RESPONSES_VISIBLE)));
    const bool responses_visible_replay_without_reasoning =
        responses_protocol &&
        j->req.responses_requires_live_reasoning &&
        !responses_reasoning_state_preserved;
    const int prompt_tokens = prompt_for_sync->len;
    /* OpenAI usage details: the reusable prefix is a cache read, while the
     * effective prompt suffix evaluated by ds4_session_sync() is written into
     * the live KV cache and can be reused by the next request. */
    j->req.cache_read_tokens = cached;
    j->req.cache_write_tokens = prompt_tokens > cached ? prompt_tokens - cached : 0;

    const double t0 = server_now_sec();
    uint64_t trace_id = trace_begin(s, j, cached, prompt_tokens, &cache_diag,
                                    cache_source, disk_cached, disk_cache_path);
    char ctx_span[48];
    request_ctx_span(ctx_span, sizeof(ctx_span), cached, prompt_tokens);
    server_prefill_progress progress = {
        .srv = s,
        .kind = j->req.kind,
        .prompt_tokens = prompt_tokens,
        .cached_tokens = cached,
        .has_tools = j->req.has_tools,
        .responses_protocol = responses_protocol,
        .t0 = t0,
        .fd = j->fd,
        .stream = j->req.stream,
        .enable_cors = s->enable_cors,
    };
    snprintf(progress.ctx, sizeof(progress.ctx), "%s", ctx_span);
    char req_flags[64];
    log_flags(req_flags, sizeof(req_flags), responses_protocol,
              j->req.has_tools, false, false, false);
    if (responses_live_continuation) {
        server_log(DS4_LOG_PREFILL,
                   "ds4-server: responses live continuation RESPPROTO match=%s ids=%d cached=%d prompt=%d",
                   responses_live_match ? responses_live_match : "unknown",
                   responses_live_match_ids,
                   cached,
                   prompt_tokens);
    } else if (anthropic_live_continuation) {
        server_log(DS4_LOG_PREFILL,
                   "ds4-server: anthropic live continuation match=tool-output-ids ids=%d cached=%d prompt=%d",
                   anthropic_live_match_ids,
                   cached,
                   prompt_tokens);
    } else if (thinking_live_continuation) {
        server_log(DS4_LOG_PREFILL,
                   "ds4-server: thinking live continuation match=visible-prefix cached=%d prompt=%d",
                   cached,
                   prompt_tokens);
    }
    if (responses_visible_replay_without_reasoning) {
        /* The request replays a prior tool-call turn but omits the hidden
         * reasoning that originally led to it.  A live Responses checkpoint, or
         * a responses-visible disk checkpoint, would preserve that hidden KV.
         * If neither is available, continue from the visible transcript instead
         * of surfacing a hard error to the user.  This is lower fidelity, but it
         * lets old / restarted agent sessions recover and is exactly what the
         * client asked us to prefill. */
        server_log(DS4_LOG_WARNING,
                   "ds4-server: responses replay RESPPROTO missing reasoning state; continuing from visible history source=%s cached=%d prompt=%d",
                   cache_source,
                   cached,
                   prompt_tokens);
        trace_event(s, trace_id,
                    "responses replay missing reasoning state; continuing from visible history source=%s cached=%d",
                    cache_source, cached);
    }
    server_log(DS4_LOG_PREFILL,
               "ds4-server: %s ctx=%s%s%s prompt start",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               ctx_span,
               req_flags[0] ? " " : "",
               req_flags);
    ds4_session_set_progress(s->session, server_progress_cb, &progress);
    ds4_session_set_display_progress(s->session, server_progress_cb, &progress);

    int cold_store_len = 0;
    if (cached == 0 &&
        s->kv.enabled &&
        prompt_for_sync->len >= s->kv.opt.min_tokens &&
        s->kv.opt.cold_max_tokens > 0 &&
        prompt_for_sync->len <= s->kv.opt.cold_max_tokens)
    {
        const int anchor = kv_cache_chat_anchor_pos(&s->kv, prompt_for_sync,
                                                    ds4_token_user(s->engine),
                                                    ds4_token_assistant(s->engine));
        cold_store_len = anchor >= s->kv.opt.min_tokens ?
                         anchor : kv_cache_store_len(&s->kv, prompt_for_sync->len);
    }
    int suppressed_continued_last = -1;
    if (cold_store_len >= s->kv.opt.min_tokens) {
        /* A cold checkpoint can land exactly on the continued-checkpoint
         * frontier.  The prefill progress callback would then write the same
         * prefix as "continued" while we are intentionally stopping there to
         * write it as "cold".  Mark the frontier as already handled before the
         * sync reaches it; if the cold write fails, restore the old schedule so
         * a later continued write can still try. */
        suppressed_continued_last =
            kv_cache_suppress_continued_store(&s->kv, cold_store_len);
    }

    if (s->kv.enabled &&
        cold_store_len >= s->kv.opt.min_tokens &&
        cold_store_len < prompt_for_sync->len)
    {
        ds4_tokens prefix = {0};
        tokens_copy_prefix(&prefix, prompt_for_sync, cold_store_len);
        if (ds4_session_sync(s->session, &prefix, err, sizeof(err)) != 0) {
            ds4_tokens_free(&prefix);
            ds4_tokens_free(&effective_prompt);
            ds4_session_set_progress(s->session, NULL, NULL);
            ds4_session_set_display_progress(s->session, NULL, NULL);
            kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                                  cold_store_len);
            kv_cache_discard_failed_disk_entry(s, disk_cache_path);
            free(disk_cache_path);
            trace_event(s, trace_id, "prefill failed: %s", err);
            send_prefill_failure_response(s, j, &progress, ctx_span, req_flags, err);
            return;
        }
        if (kv_cache_store_live_prefix(s, prompt_for_sync, cold_store_len, "cold")) {
            kv_cache_note_store(&s->kv, cold_store_len);
            suppressed_continued_last = -1;
        } else {
            kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                                  cold_store_len);
            suppressed_continued_last = -1;
        }
        ds4_tokens_free(&prefix);
    }

    if (ds4_session_sync(s->session, prompt_for_sync, err, sizeof(err)) != 0) {
        ds4_tokens_free(&effective_prompt);
        ds4_session_set_progress(s->session, NULL, NULL);
        ds4_session_set_display_progress(s->session, NULL, NULL);
        kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                              cold_store_len);
        kv_cache_discard_failed_disk_entry(s, disk_cache_path);
        free(disk_cache_path);
        trace_event(s, trace_id, "prefill failed: %s", err);
        send_prefill_failure_response(s, j, &progress, ctx_span, req_flags, err);
        return;
    }
    free(disk_cache_path);
    /* Once a non-live request wins, old protocol live bindings are stale. Keep
     * a binding only when this request explicitly continued from it. */
    if (!responses_live_continuation) responses_live_clear(s);
    if (!anthropic_live_continuation) anthropic_live_clear(s);
    if (!thinking_live_continuation) thinking_live_clear(s);
    ds4_session_set_progress(s->session, NULL, NULL);
    ds4_session_set_display_progress(s->session, NULL, NULL);
    kv_cache_maybe_store_continued(s);
    server_log(DS4_LOG_PREFILL,
               "ds4-server: %s ctx=%s%s%s prompt done %.3fs",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               ctx_span,
               req_flags[0] ? " " : "",
               req_flags,
               server_now_sec() - t0);
    if (cold_store_len == prompt_for_sync->len) {
        if (kv_cache_store_live_prefix(s, prompt_for_sync, cold_store_len, "cold")) {
            kv_cache_note_store(&s->kv, cold_store_len);
            suppressed_continued_last = -1;
        } else {
            kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                                  cold_store_len);
        }
    }
    char id[96];
    snprintf(id, sizeof(id), "%s-%llu",
             j->req.kind == REQ_CHAT ? "chatcmpl" : "cmpl",
             (unsigned long long)++s->seq);

    bool structured_stream = request_uses_structured_stream(&j->req);
    anthropic_stream anthropic_live = {0};
    openai_stream openai_live = {0};
    responses_stream responses_live = {0};
    const bool openai_live_chat = request_uses_openai_live_stream(&j->req);
    const bool responses_live_chat = request_uses_responses_live_stream(&j->req);
    long responses_created_at = (long)time(NULL);
    if (j->req.stream) {
        if (progress.stream_failed) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s%s%s stream closed during prefill",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       req_flags[0] ? " " : "",
                       req_flags);
            ds4_tokens_free(&effective_prompt);
            return;
        }
        /* The prefill progress callback may have already sent the SSE headers
         * to keep the connection alive during a long prefill. Only emit them
         * here when prefill never fired (e.g. fully cached prompt). */
        if (!progress.headers_sent && !sse_headers(j->fd, s->enable_cors)) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s%s%s sse headers failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       req_flags[0] ? " " : "",
                       req_flags);
            ds4_tokens_free(&effective_prompt);
            return;
        }
        progress.headers_sent = true;
        if (j->req.api == API_ANTHROPIC &&
            !anthropic_sse_start_live(j->fd, &j->req, id,
                                      prompt_tokens, &anthropic_live)) {
            server_log(DS4_LOG_GENERATION, "ds4-server: chat ctx=%s anthropic stream start failed", ctx_span);
            ds4_tokens_free(&effective_prompt);
            return;
        }
        if (j->req.api == API_OPENAI && j->req.kind == REQ_CHAT &&
            !sse_chunk(j->fd, &j->req, id, NULL, NULL)) {
            server_log(DS4_LOG_GENERATION, "ds4-server: chat ctx=%s openai role chunk failed", ctx_span);
            ds4_tokens_free(&effective_prompt);
            return;
        }
        if (openai_live_chat) openai_stream_start(&j->req, &openai_live);
        if (responses_live_chat) {
            responses_stream_init(&j->req, &responses_live);
            responses_live.active = true;
            if (!responses_sse_created(j->fd, &j->req, &responses_live, responses_created_at)) {
                server_log(DS4_LOG_GENERATION,
                           "ds4-server: chat ctx=%s%s%s responses created event failed",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags);
                responses_stream_free(&responses_live);
                ds4_tokens_free(&effective_prompt);
                return;
            }
        }
    }

    bool dsml_recovery_attempted = false;
    uint64_t rng = j->req.seed ? j->req.seed :
        (((uint64_t)time(NULL) << 32) ^ ((uint64_t)s->seq << 1) ^ (uint64_t)(uintptr_t)j);
decode_again:
    ;
    buf text = {0};
    size_t plain_stream_pos = 0;
    size_t stop_scan_from = 0;
    const char *finish = "length";
    int completion = 0;
    int max_tokens = j->req.max_tokens;
    int room = ds4_session_ctx(s->session) - ds4_session_pos(s->session);
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    bool saw_orphan_tool_end = false;
    size_t tool_scan_from = 0;
    int next_tool_progress = 128;
    int next_decode_log = 50;
    if (max_tokens < 0) max_tokens = 0;
    if (max_tokens > room) max_tokens = room;
    trace_event(s, trace_id, "prefill done; decode_max=%d ctx_room=%d", max_tokens, room);
    const double decode_t0 = server_now_sec();
    double last_decode_log_t = decode_t0;
    int last_decode_log_completion = 0;
    thinking_state thinking = thinking_state_from_prompt(&j->req);
    const bool thinking_gates_tool_markers = ds4_think_mode_enabled(j->req.think_mode);
    bool tool_scan_waiting_for_think_close =
        thinking_gates_tool_markers && thinking.inside;
    size_t think_recovery_scan_from = 0;
    const bool think_tool_recovery_enabled =
        getenv("DS4_SERVER_DISABLE_THINK_TOOL_RECOVERY") == NULL;
    dsml_decode_tracker dsml_tracker;
    dsml_decode_tracker_init(&dsml_tracker);

    while (!g_stop_requested && completion < max_tokens &&
           ds4_session_pos(s->session) < ds4_session_ctx(s->session)) {
        dsml_decode_state dsml_state = j->req.kind == REQ_CHAT && j->req.has_tools ?
            dsml_tracker.decode : DSML_DECODE_OUTSIDE;
        const bool in_tool_call = dsml_decode_state_is_tool(dsml_state);
        if (!(j->req.kind == REQ_CHAT && j->req.has_tools && (saw_tool_start || in_tool_call))) {
            kv_cache_maybe_store_continued(s);
        }
        float temperature = j->req.temperature;
        int top_k = j->req.top_k;
        float top_p = j->req.top_p;
        float min_p = j->req.min_p;
        /* Thinking mode normally forces sampling defaults for quality, but an
         * EXPLICIT temperature==0 (default is 1.0) means the caller wants greedy
         * decode — honor it so MTP speculative decode (greedy-only) can engage. */
        if (ds4_think_mode_enabled(j->req.think_mode) && j->req.temperature > 0.0f) {
            temperature = DS4_DEFAULT_TEMPERATURE;
            top_k = 0;
            top_p = DS4_DEFAULT_TOP_P;
            min_p = DS4_DEFAULT_MIN_P;
        }
        if (in_tool_call && !dsml_decode_state_uses_payload_sampling(dsml_state)) {
            temperature = 0.0f;
        }
        int token = ds4_session_sample(s->session, temperature, top_k, top_p, min_p, &rng);
        if (token == ds4_token_eos(s->engine)) {
            finish = "stop";
            break;
        }

        int toks[17];
        int ntok = 0;
        if (temperature <= 0.0f &&
            ds4_engine_has_dspark(s->engine) &&
            getenv("DS4_DSPARK_DISABLE") == NULL)
        {
            ntok = ds4_session_eval_speculative_block(s->session,
                                                      token,
                                                      max_tokens - completion,
                                                      ds4_token_eos(s->engine),
                                                      toks,
                                                      (int)(sizeof(toks) / sizeof(toks[0])),
                                                      err,
                                                      sizeof(err));
            if (ntok < 0) {
                finish = "error";
                break;
            }
        } else if (temperature <= 0.0f &&
            ds4_engine_mtp_draft_tokens(s->engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL)
        {
            ntok = ds4_session_eval_speculative_argmax(s->session,
                                                       token,
                                                       max_tokens - completion,
                                                       ds4_token_eos(s->engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                finish = "error";
                break;
            }
        } else {
            if (ds4_session_eval(s->session, token, err, sizeof(err)) != 0) {
                finish = "error";
                break;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop_decode = false;
        for (int ti = 0; ti < ntok && completion < max_tokens; ti++) {
            token = toks[ti];
            if (token == ds4_token_eos(s->engine)) {
                finish = "stop";
                stop_decode = true;
                break;
            }

            size_t piece_len = 0;
            char *piece = ds4_token_text(s->engine, token, &piece_len);
            completion++;

            trace_piece(s, trace_id, piece, piece_len);
            buf_append(&text, piece, piece_len);
            thinking_state_feed(&thinking, piece, piece_len);
            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                dsml_decode_tracker_update(&dsml_tracker, text.ptr, text.len);
            }

            size_t stop_pos = 0, stop_len = 0;
            bool hit_stop = stop_list_find_from(&j->req.stops, text.ptr,
                                                stop_scan_from,
                                                &stop_pos, &stop_len);
            size_t stream_len = hit_stop ?
                stop_pos : stop_list_stream_safe_len(&j->req.stops, text.len);
            if (stream_len > text.len) stream_len = text.len;
            stream_len = utf8_stream_safe_len(text.ptr, plain_stream_pos,
                                              stream_len, hit_stop);
            if (!hit_stop && j->req.stops.max_len > 1) {
                const size_t hold = j->req.stops.max_len - 1;
                stop_scan_from = text.len > hold ? text.len - hold : 0;
            }

            if (j->req.stream && !structured_stream && stream_len > plain_stream_pos) {
                char *delta = xstrndup(text.ptr + plain_stream_pos, stream_len - plain_stream_pos);
                bool ok = sse_chunk(j->fd, &j->req, id, delta, NULL);
                free(delta);
                if (!ok) {
                    finish = "error";
                    snprintf(err, sizeof(err), "client stream write failed");
                    free(piece);
                    stop_decode = true;
                    break;
                }
                plain_stream_pos = stream_len;
            }
            if (j->req.stream && j->req.api == API_ANTHROPIC &&
                !anthropic_sse_stream_update(j->fd, s, &j->req, id,
                                             &anthropic_live, text.ptr, stream_len,
                                             false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            if (openai_live_chat &&
                !openai_sse_stream_update(j->fd, s, &j->req, id,
                                          &openai_live, text.ptr, stream_len,
                                          false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            if (responses_live_chat &&
                !responses_sse_stream_update(j->fd, &j->req,
                                             &responses_live, text.ptr, stream_len,
                                             false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            free(piece);

            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                if (thinking_gates_tool_markers && thinking.inside) {
                    /* A DSML block inside reasoning is not executable.  This is
                     * the live guard: do not let a quoted or mistaken marker in
                     * <think> stop decoding as a real tool call.  A complete
                     * stanza opening, however, almost always means the model
                     * forgot to close its thinking; recover by forcing the
                     * close so the model restarts the call on the executable
                     * side. */
                    const int recovered = think_tool_recovery_enabled ?
                        chat_think_tool_recovery(s, &text, &thinking,
                                                 &think_recovery_scan_from,
                                                 &completion, max_tokens,
                                                 err, sizeof(err)) : 0;
                    if (recovered < 0) {
                        finish = "error";
                        stop_decode = true;
                        break;
                    }
                    if (recovered) {
                        server_log(DS4_LOG_WARNING,
                                   "ds4-server: chat ctx=%s%s%s tool call inside unclosed <think>; "
                                   "forced </think> after %d generated tokens",
                                   ctx_span,
                                   req_flags[0] ? " " : "",
                                   req_flags,
                                   completion);
                        trace_event(s, trace_id,
                                    "think tool recovery after %d generated tokens",
                                    completion);
                        dsml_decode_tracker_update(&dsml_tracker, text.ptr, text.len);
                        tool_scan_waiting_for_think_close = true;
                    } else {
                        tool_scan_waiting_for_think_close = true;
                        tool_scan_from = text.len;
                    }
                } else {
                    if (tool_scan_waiting_for_think_close) {
                        const char *think_end = find_last_substr(text.ptr, "</think>");
                        tool_scan_from = think_end ? (size_t)((think_end + 8) - text.ptr) : text.len;
                        if (tool_scan_from > text.len) tool_scan_from = text.len;
                        tool_scan_waiting_for_think_close = false;
                    }
                    if (tool_scan_from > text.len) tool_scan_from = text.len;
                    const char *tool_scan = text.ptr ? text.ptr + tool_scan_from : "";
                    bool orphan_end = false;
                    bool old_start = saw_tool_start;
                    bool old_end = saw_tool_end;
                    observe_tool_markers(tool_scan, &saw_tool_start, &saw_tool_end, &orphan_end);
                    if (orphan_end && !saw_orphan_tool_end) {
                        saw_orphan_tool_end = true;
                        server_log(DS4_LOG_WARNING,
                                   "ds4-server: chat ctx=%s%s%s ignored orphan tool-call end marker after %d generated tokens",
                                   ctx_span,
                                   req_flags[0] ? " " : "",
                                   req_flags,
                                   completion);
                        trace_event(s, trace_id,
                                    "ignored orphan tool-call end marker after %d generated tokens",
                                    completion);
                    }
                    if (saw_tool_start && !old_start) {
                        trace_event(s, trace_id, "entered tool-call block after %d generated tokens", completion);
                    }
                    if (saw_tool_end && !old_end) {
                        trace_event(s, trace_id, "closed tool-call block after %d generated tokens", completion);
                    }
                    const size_t marker_hold = 80;
                    size_t hold_from = text.len > marker_hold ? text.len - marker_hold : 0;
                    if (hold_from > tool_scan_from) tool_scan_from = hold_from;
                    if (s->trace && completion >= next_tool_progress) {
                        trace_event(s, trace_id,
                                    "progress gen=%d dsml_start=%d dsml_end=%d",
                                    completion, saw_tool_start ? 1 : 0, saw_tool_end ? 1 : 0);
                        next_tool_progress += 128;
                    }
                }
            }

            if (completion >= next_decode_log) {
                log_decode_progress(j->req.kind, prompt_tokens, completion,
                                    responses_protocol,
                                    j->req.has_tools,
                                    thinking.inside,
                                    saw_tool_start,
                                    saw_tool_end,
                                    decode_t0,
                                    &last_decode_log_t,
                                    &last_decode_log_completion);
                next_decode_log += 50;
            }

            if (hit_stop) {
                (void)stop_len;
                finish = "stop";
                text.len = stop_pos;
                text.ptr[text.len] = '\0';
                ds4_session_invalidate(s->session);
                stop_decode = true;
                break;
            }

            if (j->req.kind == REQ_CHAT && j->req.has_tools && saw_tool_end) {
                finish = "tool_calls";
                stop_decode = true;
                break;
            }
        }
        if (stop_decode) break;
    }

    if (g_stop_requested && strcmp(finish, "error") != 0) {
        finish = "error";
        snprintf(err, sizeof(err), "shutdown requested");
    }

    if (j->req.kind == REQ_CHAT && j->req.has_tools &&
        saw_tool_start && !saw_tool_end && strcmp(finish, "error") != 0)
    {
        /* Deterministically complete a simple truncation.  Anything more than
         * missing closing tags stays model-owned: for non-streaming requests,
         * append a tool error plus prompt reminder to the live session and let
         * the model issue a fresh call. */
        bool completed_truncation = false;
        buf repaired = {0};
        if (try_repair_dsml(text.ptr, text.len, &repaired)) {
            /* Parse repaired text to verify it produces valid tool calls */
            tool_calls test_calls = {0};
            char *test_content = NULL;
            char *test_reasoning = NULL;
            bool repair_ok = parse_generated_message_ex(repaired.ptr, false, &test_content, &test_reasoning, &test_calls);
            free(test_content);
            free(test_reasoning);
            if (repair_ok && test_calls.len > 0) {
                /* Repair succeeded - replace text with repaired version */
                free(text.ptr);
                text.ptr = buf_take(&repaired);
                text.len = strlen(text.ptr);
                saw_tool_end = true;
                completed_truncation = true;
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s repaired unterminated tool call (%d calls recovered)",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags,
                           test_calls.len);
                trace_event(s, trace_id, "repaired unterminated tool call (%d calls recovered)", test_calls.len);
            }
            tool_calls_free(&test_calls);
        }
        if (!completed_truncation) {
            if (!j->req.stream && !dsml_recovery_attempted) {
                int recovery_tokens = 0;
                char recovery_err[160] = {0};
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s unterminated tool call; continuing with model-visible tool error",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags);
                trace_event(s, trace_id,
                            "unterminated tool call; continuing with model-visible tool error");
                if (continue_after_invalid_dsml(s, &j->req, &thinking,
                                                "unterminated tool call",
                                                &recovery_tokens,
                                                recovery_err,
                                                sizeof(recovery_err)))
                {
                    dsml_recovery_attempted = true;
                    server_log(DS4_LOG_GENERATION,
                               "ds4-server: chat ctx=%s%s%s tool-error continuation appended %d tokens",
                               ctx_span,
                               req_flags[0] ? " " : "",
                               req_flags,
                               recovery_tokens);
                    trace_event(s, trace_id,
                                "tool-error continuation appended %d tokens",
                                recovery_tokens);
                    buf_free(&repaired);
                    buf_free(&text);
                    goto decode_again;
                }
                finish = "error";
                snprintf(err, sizeof(err), "invalid tool call recovery failed: %s",
                         recovery_err[0] ? recovery_err : "unknown error");
            } else {
                finish = "error";
                snprintf(err, sizeof(err), "unterminated tool call");
            }
        }
        buf_free(&repaired);
    }

    if (completion > last_decode_log_completion) {
        log_decode_progress(j->req.kind, prompt_tokens, completion,
                            responses_protocol,
                            j->req.has_tools,
                            thinking.inside,
                            saw_tool_start,
                            saw_tool_end,
                            decode_t0,
                            &last_decode_log_t,
                            &last_decode_log_completion);
    }

    if (j->req.stream && !structured_stream && text.len > plain_stream_pos) {
        char *tail = xstrndup(text.ptr + plain_stream_pos, text.len - plain_stream_pos);
        if (!sse_chunk(j->fd, &j->req, id, tail, NULL)) finish = "error";
        free(tail);
    }

    tool_calls parsed_calls = {0};
    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    const char *final_finish = finish;
    bool recovered_tool_parse_failure = false;
    if (j->req.kind == REQ_CHAT) {
        bool parsed_ok = parse_generated_message_for_response(
            text.ptr ? text.ptr : "",
            j->req.has_tools,
            saw_tool_start,
            ds4_think_mode_enabled(j->req.think_mode),
            &final_finish,
            err,
            sizeof(err),
            &parsed_content,
            &parsed_reasoning,
            &parsed_calls,
            &recovered_tool_parse_failure);
        if (!parsed_ok && recovered_tool_parse_failure && j->req.has_tools && saw_tool_start) {
            /* parse_generated_message failed even though DSML was present.
             * Semantic repair is intentionally avoided: if the parser cannot
             * execute the block, feed the model a tool error and the protocol
             * reminder so it owns the corrected next action. */
            if (!j->req.stream && !dsml_recovery_attempted) {
                int recovery_tokens = 0;
                char recovery_err[160] = {0};
                const char *detail = err[0] ? err : "invalid tool call";
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s invalid tool call; continuing with model-visible tool error",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags);
                trace_event(s, trace_id,
                            "invalid tool call; continuing with model-visible tool error");
                if (continue_after_invalid_dsml(s, &j->req, &thinking,
                                                detail,
                                                &recovery_tokens,
                                                recovery_err,
                                                sizeof(recovery_err)))
                {
                    dsml_recovery_attempted = true;
                    server_log(DS4_LOG_GENERATION,
                               "ds4-server: chat ctx=%s%s%s tool-error continuation appended %d tokens",
                               ctx_span,
                               req_flags[0] ? " " : "",
                               req_flags,
                               recovery_tokens);
                    trace_event(s, trace_id,
                                "tool-error continuation appended %d tokens",
                                recovery_tokens);
                    free(parsed_content);
                    free(parsed_reasoning);
                    tool_calls_free(&parsed_calls);
                    buf_free(&text);
                    goto decode_again;
                }
                final_finish = "error";
                snprintf(err, sizeof(err), "invalid tool call recovery failed: %s",
                         recovery_err[0] ? recovery_err : "unknown error");
            }
            if (!parsed_ok) {
                /* Print raw DSML snippet for debugging */
                size_t dsml_snippet_len = 0;
                const char *dsml_start = NULL;
                const char *p;
                for (p = text.ptr; p && (size_t)(p - text.ptr) < text.len - 20; p++) {
                    if ((strncmp(p, DS4_TOOL_CALLS_START, strlen(DS4_TOOL_CALLS_START)) == 0) ||
                        (strncmp(p, DS4_TOOL_CALLS_START_SHORT, strlen(DS4_TOOL_CALLS_START_SHORT)) == 0) ||
                        (strncmp(p, "<tool_calls>", 12) == 0)) {
                        dsml_start = p;
                        break;
                    }
                }
                if (dsml_start) {
                    dsml_snippet_len = text.len - (dsml_start - text.ptr);
                    if (dsml_snippet_len > 500) dsml_snippet_len = 500;
                }
                /* Also log a snippet of the full text to see what the model output */
                size_t text_snippet_len = text.len > 300 ? 300 : text.len;
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s invalid tool call returned as assistant text finish=%s [text_len=%zu saw_start=%d saw_end=%d text_snippet: %.*s]",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags,
                           final_finish,
                           text.len,
                           saw_tool_start,
                           saw_tool_end,
                           (int)text_snippet_len,
                           text.ptr ? text.ptr : "(null)");
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s invalid tool call dsml_snippet: %.*s",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags,
                           (int)dsml_snippet_len,
                           dsml_start ? dsml_start : "(none)");
                trace_event(s, trace_id,
                            "invalid tool call returned as assistant text finish=%s",
                            final_finish);
            }
        }
        if (parsed_calls.len) {
            if (openai_live_chat) apply_openai_stream_tool_ids(&parsed_calls, &openai_live);
            if (j->req.api == API_ANTHROPIC && j->req.stream)
                apply_anthropic_stream_tool_ids(&parsed_calls, &anthropic_live);
            assign_tool_call_ids(s, &parsed_calls, j->req.api);
            tool_memory_remember(s, &parsed_calls);
            final_finish = "tool_calls";
        } else if (j->req.api == API_RESPONSES) {
            responses_live_clear(s);
        }
    }
    log_tool_calls_summary(ctx_span, &parsed_calls,
                           responses_protocol);

    trace_finish(s, trace_id, &j->req, final_finish, completion,
                 saw_tool_start, saw_tool_end,
                 parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                 parsed_reasoning, &parsed_calls, server_now_sec() - t0);

    if (j->req.api == API_RESPONSES) {
        if (strcmp(final_finish, "error") && strcmp(final_finish, "length")) {
            /* Store the post-turn visible transcript plus the live token
             * frontier.  The next Responses request may replay only this
             * visible surface, while the real session also contains hidden
             * reasoning and exact sampled tool-call bytes. */
            char *visible_suffix =
                build_responses_visible_assistant_suffix(&j->req,
                    parsed_content ? parsed_content : "",
                    parsed_reasoning,
                    &parsed_calls);
            buf visible = {0};
            buf_puts(&visible, j->req.prompt_text ? j->req.prompt_text : "");
            buf_puts(&visible, visible_suffix ? visible_suffix : "");
            responses_live_remember(s, visible.ptr ? visible.ptr : "",
                                    parsed_calls.len ? &parsed_calls : NULL);
            buf_free(&visible);
            free(visible_suffix);
        } else {
            responses_live_clear(s);
        }
    }
    if (j->req.api == API_ANTHROPIC) {
        if (parsed_calls.len && strcmp(final_finish, "error") &&
            strcmp(final_finish, "length"))
        {
            anthropic_live_remember(s, &parsed_calls);
        } else {
            anthropic_live_clear(s);
        }
    }

    if (j->req.kind == REQ_CHAT && parsed_calls.len &&
        j->req.api != API_RESPONSES &&
        should_canonicalize_tool_checkpoint(s, &parsed_calls))
    {
        /* Chat/completions has no protocol object that binds the next request
         * to this live KV state.  Canonicalize only the fallback tool-call
         * path where we lack exact sampled DSML replay; when raw DSML is known,
         * replaying those bytes keeps future prompts aligned without rebuilding
         * hidden reasoning.  Responses deliberately skips this path because its
         * previous_response_id contract binds the next turn to live state. */
        canonicalize_tool_checkpoint(s, j, ctx_span, trace_id,
                                     parsed_content ? parsed_content : "",
                                     parsed_reasoning, &parsed_calls);
        thinking_live_clear(s);
    } else if (parsed_calls.len) {
        thinking_live_clear(s);
    } else if (!parsed_calls.len &&
               should_remember_thinking_checkpoint(&j->req, &thinking, final_finish)) {
        remember_thinking_checkpoint(s, j, ctx_span, trace_id,
                                     parsed_content ? parsed_content : "");
    } else if (!parsed_calls.len) {
        thinking_live_clear(s);
    }

    if (j->req.stream) {
        bool response_ok = true;
        if (j->req.api == API_ANTHROPIC) {
            response_ok = anthropic_sse_finish_live(j->fd, s, &j->req, id, &anthropic_live,
                                                    text.ptr ? text.ptr : "", text.len,
                                                    &parsed_calls, final_finish, completion);
        } else if (openai_live_chat) {
            response_ok = openai_sse_finish_live(j->fd, s, &j->req, id, &openai_live,
                                                 text.ptr ? text.ptr : "", text.len,
                                                 &parsed_calls, final_finish,
                                                 prompt_tokens, completion);
        } else if (responses_live_chat) {
            /* If parse recovered a malformed tool call back to plain text,
             * pass parsed_content so the streaming tail can be flushed; in
             * the normal path parsed_content is the assistant text we already
             * streamed and the diff is empty. */
            const char *recover =
                recovered_tool_parse_failure ? parsed_content : NULL;
            response_ok = responses_sse_finish_live(j->fd, &j->req, &responses_live,
                                                    text.ptr ? text.ptr : "", text.len,
                                                    recover,
                                                    &parsed_calls, final_finish,
                                                    prompt_tokens, completion,
                                                    responses_created_at);
        } else if (structured_stream) {
            response_ok = sse_chat_finish(j->fd, &j->req, id,
                                          parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                                          parsed_reasoning,
                                          &parsed_calls, final_finish,
                                          prompt_tokens, completion);
        } else {
            response_ok = sse_chunk(j->fd, &j->req, id, NULL, final_finish) &&
                          sse_done(j->fd, &j->req, id, prompt_tokens, completion);
        }
        if (!response_ok) {
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: %s ctx=%s%s%s final stream failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       req_flags[0] ? " " : "",
                       req_flags);
        }
    } else if (j->req.api == API_ANTHROPIC) {
        anthropic_final_response(j->fd, s->enable_cors, &j->req, id,
                                 parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 prompt_tokens, completion);
    } else if (j->req.api == API_RESPONSES) {
        responses_final_response(j->fd, s->enable_cors, &j->req, id,
                                 parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 prompt_tokens, completion);
    } else {
        final_response(j->fd, s->enable_cors, &j->req, id,
                       parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                       parsed_reasoning,
                       &parsed_calls, final_finish,
                       prompt_tokens, completion);
    }
    if (j->req.kind == REQ_CHAT && j->req.has_tools) {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  responses_protocol,
                  true,
                  thinking.inside,
                  saw_tool_start,
                  saw_tool_end);
        if (!strcmp(final_finish, "error") && err[0]) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: chat ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       err,
                       server_now_sec() - t0);
        } else {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: chat ctx=%s gen=%d%s%s finish=%s %.3fs",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       server_now_sec() - t0);
        }
    } else {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  responses_protocol,
                  j->req.has_tools,
                  thinking.inside,
                  false,
                  false);
        if (!strcmp(final_finish, "error") && err[0]) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       err,
                       server_now_sec() - t0);
        } else {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s gen=%d%s%s finish=%s %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       server_now_sec() - t0);
        }
    }
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&parsed_calls);
    anthropic_stream_free(&anthropic_live);
    openai_stream_free(&openai_live);
    responses_stream_free(&responses_live);
    buf_free(&text);
    ds4_tokens_free(&effective_prompt);
}



bool enqueue(server *s, job *j) {
    pthread_mutex_lock(&s->mu);
    if (s->stopping) {
        pthread_mutex_unlock(&s->mu);
        return false;
    }
    if (s->tail) s->tail->next = j; else s->head = j;
    s->tail = j;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
    return true;
}



static job *dequeue(server *s) {
    pthread_mutex_lock(&s->mu);
    while (!s->head && !s->stopping) pthread_cond_wait(&s->cv, &s->mu);
    if (!s->head) {
        pthread_mutex_unlock(&s->mu);
        return NULL;
    }
    job *j = s->head;
    s->head = j->next;
    if (!s->head) s->tail = NULL;
    pthread_mutex_unlock(&s->mu);
    j->next = NULL;
    return j;
}



void *worker_main(void *arg) {
    server *s = arg;
    for (;;) {
        job *j = dequeue(s);
        if (!j) break;
        generate_job(s, j);
        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);
    }
    return NULL;
}

