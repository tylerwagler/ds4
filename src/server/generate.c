#include "ds4_server_internal.h"
/* ds4_gpu_tensor_alloc_bytes_current: the eviction path verifies that freeing
 * a session releases (allocator ground truth) what the ledger committed. */
#include "ds4_gpu.h"



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
                                    session_slot *sl,
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

    const int room = ds4_session_ctx(sl->sess) - ds4_session_pos(sl->sess);
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
        if (ds4_session_eval(sl->sess, toks.v[i], err, errlen) != 0) {
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



static bool append_rendered_suffix_to_live_session(server *s, session_slot *sl,
                                                   const char *suffix,
                                                   int *tokens_appended,
                                                   char *err, size_t errlen) {
    if (tokens_appended) *tokens_appended = 0;
    if (!s || !suffix || !suffix[0]) return true;
    const ds4_tokens *live = ds4_session_tokens(sl->sess);
    if (!live) {
        if (err && errlen) snprintf(err, errlen, "live session is unavailable");
        return false;
    }

    ds4_tokens target = {0};
    build_prompt_from_exact_prefix_and_text_suffix(s->engine, live, suffix, &target);
    const int before = ds4_session_pos(sl->sess);
    bool ok = ds4_session_sync(sl->sess, &target, err, errlen) == 0;
    if (ok && tokens_appended) {
        int delta = ds4_session_pos(sl->sess) - before;
        *tokens_appended = delta > 0 ? delta : 0;
    }
    ds4_tokens_free(&target);
    return ok;
}



static bool continue_after_invalid_dsml(server *s, session_slot *sl,
                                        const request *r,
                                        const thinking_state *thinking,
                                        const char *detail,
                                        int *tokens_appended,
                                        char *err, size_t errlen) {
    char *suffix = build_invalid_dsml_tool_error_suffix(r, thinking, detail);
    bool ok = append_rendered_suffix_to_live_session(s, sl, suffix,
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
        if (p->srv && p->slot && current > p->cached_tokens) {
            kv_cache_maybe_store_continued(p->srv, p->slot);
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
    if (p->srv && p->slot && current > p->cached_tokens) {
        kv_cache_maybe_store_continued(p->srv, p->slot);
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



static void remember_thinking_checkpoint(server *s, session_slot *sl,
                                         const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content) {
    char *visible = build_toolless_thinking_visible_text(&j->req, content);
    if (!visible) return;

    thinking_live_remember(s, sl, visible);
    server_log(DS4_LOG_KVCACHE,
               "ds4-server: thinking live checkpoint remembered ctx=%s live=%d visible=%zu",
               ctx, ds4_session_pos(sl->sess), strlen(visible));
    trace_event(s, trace_id,
                "thinking live checkpoint remembered: live=%d visible=%zu",
                ds4_session_pos(sl->sess), strlen(visible));
    free(visible);
}

/* Tool-call finish WITH thinking on: the model emitted <think>reasoning</think>
 * before the DSML tool call, so the reasoning tokens sit in the live KV. The
 * client replays this turn with the reasoning stripped (<think></think>), so the
 * exact-DSML-replay shortcut no longer aligns and every follow-up re-prefills the
 * whole conversation. Fix it like the toolless thinking path: remember the
 * thinking-STRIPPED bytes the next request will render as a key, but keep the
 * live tokens (reasoning included) as the sampled frontier. The next request then
 * byte-matches the key and continues from live KV — no rewrite, no rebuild. */
static void remember_tool_thinking_checkpoint(server *s, session_slot *sl,
                                              const job *j, const char *ctx,
                                              uint64_t trace_id, const char *content,
                                              const tool_calls *calls) {
    if (!calls || calls->len == 0 || !j->req.prompt_text) return;
    if (!ds4_think_mode_enabled(j->req.think_mode)) return;

    /* Visible key = prompt_text (ends "<｜Assistant｜><think>") + empty-reasoning
     * suffix "</think>{content}{DSML}<EOS>" — exactly what render_chat_prompt_text
     * emits for this tool turn on the next request (reasoning dropped). */
    char *suffix = build_tool_checkpoint_suffix(&j->req, content, "", calls);
    buf visible = {0};
    buf_puts(&visible, j->req.prompt_text);
    buf_puts(&visible, suffix);
    if (visible.ptr) {
        thinking_live_remember(s, sl, visible.ptr);
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool thinking checkpoint remembered ctx=%s live=%d visible=%zu",
                   ctx, ds4_session_pos(sl->sess), visible.len);
        trace_event(s, trace_id,
                    "tool thinking checkpoint remembered: live=%d visible=%zu",
                    ds4_session_pos(sl->sess), visible.len);
    }
    free(suffix);
    buf_free(&visible);
}



/* After a successful tool-call finish, make the live checkpoint match what the
 * next request will render.  Usually that is just the exact DSML remembered by
 * tool id.  If a client sends a tool call without an id we know, the fallback
 * renderer still builds valid DSML from JSON, and this function either rewrites
 * the short suffix in place or reloads an older disk checkpoint before replay. */
static void canonicalize_tool_checkpoint(server *s, session_slot *sl,
                                         const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content,
                                         const char *reasoning, const tool_calls *calls) {
    if (!calls || calls->len == 0 || !j->req.prompt_text) return;

    char *suffix_text = build_tool_checkpoint_suffix(&j->req, content, reasoning, calls);

    buf rendered = {0};
    buf_puts(&rendered, j->req.prompt_text);
    buf_puts(&rendered, suffix_text);

    ds4_tokens canonical = {0};
    ds4_tokenize_rendered_chat(s->engine, rendered.ptr ? rendered.ptr : "", &canonical);
    const int live_len = ds4_session_pos(sl->sess);
    const int common = ds4_session_common_prefix(sl->sess, &canonical);
    if (common == live_len && canonical.len == live_len) goto done;

    size_t live_text_len = 0;
    char *live_text = render_tokens_text(s->engine, ds4_session_tokens(sl->sess), &live_text_len);
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
        ds4_session_rewrite_from_common(sl->sess, &canonical, common,
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
        int loaded = kv_cache_try_load_text(s, sl, rendered.ptr ? rendered.ptr : "",
                                            &effective, &path, NULL, false);
        if (loaded == 0) ds4_session_invalidate(sl->sess);

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
            .slot = sl,
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
        ds4_session_set_progress(sl->sess, server_progress_cb, &rebuild_progress);
        ds4_session_set_display_progress(sl->sess, server_progress_cb, &rebuild_progress);
        if (ds4_session_sync(sl->sess, sync_prompt, sync_err, sizeof(sync_err)) == 0) {
            ds4_session_set_progress(sl->sess, NULL, NULL);
            ds4_session_set_display_progress(sl->sess, NULL, NULL);
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
            ds4_session_set_progress(sl->sess, NULL, NULL);
            ds4_session_set_display_progress(sl->sess, NULL, NULL);
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



/* =========================================================================
 * Resumable per-slot generation (multi-session increment 2).
 * =========================================================================
 *
 * The old run-to-completion generate_job() is restructured into a state
 * machine the worker steps in bounded quanta:
 *
 *   GEN_PREFILL_COLD -> GEN_PREFILL_MAIN -> GEN_DECODE_INIT -> GEN_DECODE
 *        (one engine chunk per quantum)         (K tokens per quantum)
 *                                                       |
 *                              GEN_DONE <- GEN_FINISH <-+
 *                                              |
 *                (tool-error recovery, the old goto decode_again)
 *                                              v
 *                                       GEN_DECODE_INIT
 *
 * Everything that must survive across quanta lives in gen_state, hung off the
 * slot. A prefill quantum uses the engine's own chunk boundaries: a cancel
 * callback interrupts ds4_session_sync() after one completed chunk (only when
 * ds4_session_prefill_quantum_min_suffix() says resumption is bit-exact) and
 * the next quantum re-issues the sync, which resumes from the checkpoint.
 * A decode quantum runs the sampling loop for at most
 * DS4_SERVER_DECODE_QUANTUM_TOKENS tokens. Between quanta the session is not
 * touched, so the spec-decode carry (spec_carry_* in session.c) and sampling
 * rng stay valid; with one slot the quanta run back-to-back and the output is
 * byte-identical to the old function.
 *
 * Bounded exceptions that INTENTIONALLY stay run-to-completion inside one
 * quantum (kept in increment 3's scheduler by design): the tool-error
 * recovery syncs (short model-visible suffix append) and
 * canonicalize_tool_checkpoint's rebuild in GEN_FINISH. Both are rare repair
 * paths that must observe a consistent session frontier; a co-scheduled slot
 * simply waits out the (bounded) repair. The final logits-writing prefill
 * chunk likewise always completes within its quantum — the engine's cancel
 * check only interrupts when enough suffix remains for a bit-exact resume
 * (ds4_session_prefill_quantum_min_suffix).
 *
 * The largest quantum overshoot in the system is none of the above: it is
 * lazy slot provisioning (provision_slot in the scheduler below), whose
 * ds4_session_create is a multi-GiB allocation that can take SECONDS and
 * stalls every bound slot for its duration — larger than the DSpark fused
 * step's ≤17-token burst. Deliberate: all GPU work stays on this one thread
 * (CUDA-state audit, ds4_server_internal.h).
 */

typedef enum {
    GEN_PREFILL_COLD = 0, /* syncing the cold-store prefix, one chunk/quantum */
    GEN_PREFILL_MAIN,     /* syncing the effective prompt, one chunk/quantum */
    GEN_DECODE_INIT,      /* (re)initialize a decode attempt (old decode_again) */
    GEN_DECODE,           /* sampling loop, K tokens per quantum */
    GEN_FINISH,           /* parse, checkpoints, final response */
    GEN_DONE,
} gen_phase;

struct gen_state {
    job *j;
    gen_phase phase;

    /* prompt/cache resolution (owned by gen_begin, read by later phases) */
    char err[160];
    ds4_tokens effective_prompt;
    const ds4_tokens *prompt_for_sync; /* &j->req.prompt or &effective_prompt */
    bool responses_protocol;
    bool responses_live_continuation;
    bool anthropic_live_continuation;
    bool thinking_live_continuation;
    char *disk_cache_path;
    int prompt_tokens;
    double t0;
    uint64_t trace_id;
    char ctx_span[48];
    char req_flags[64];
    server_prefill_progress progress; /* stable address: callback userdata */
    int cold_store_len;
    int suppressed_continued_last;
    ds4_tokens cold_prefix;

    /* prefill quantum policy (see gen_prefill_cancel_cb) */
    uint32_t prefill_min_suffix; /* 0 = interrupting is never exact */
    int prefill_chunks_done;     /* chunks completed in the current sync call */
    int prefill_last_current;
    int prefill_total;

    /* response identity + per-protocol stream projections; these live across
     * quanta AND across decode_again recovery attempts */
    char id[96];
    bool structured_stream;
    anthropic_stream anthropic_live;
    openai_stream openai_live;
    responses_stream responses_live;
    bool openai_live_chat;
    bool responses_live_chat;
    long responses_created_at;
    bool dsml_recovery_attempted;
    uint64_t rng;

    /* decode attempt state (reset by GEN_DECODE_INIT) */
    buf text;
    size_t plain_stream_pos;
    size_t stop_scan_from;
    const char *finish;
    int completion;
    int max_tokens;
    bool saw_tool_start;
    bool saw_tool_end;
    bool saw_orphan_tool_end;
    size_t tool_scan_from;
    int next_tool_progress;
    int next_decode_log;
    double decode_t0;
    double last_decode_log_t;
    int last_decode_log_completion;
    thinking_state thinking;
    bool thinking_gates_tool_markers;
    bool tool_scan_waiting_for_think_close;
    size_t think_recovery_scan_from;
    bool think_tool_recovery_enabled;
    bool dspark_spec_enabled;
    dsml_decode_tracker dsml_tracker;

    /* deferred, non-blocking client writes (installed for send_all) */
    slot_writer writer;
};

static void gen_stream_begin(server *s, session_slot *sl);



/* Chunk-note wrapper around server_progress_cb: counts completed prefill
 * chunks in the CURRENT ds4_session_sync call so the cancel callback can
 * interrupt after exactly one chunk. Counters are reset before each sync. */
static void gen_prefill_progress_cb(void *ud, const char *event, int current, int total) {
    gen_state *g = ud;
    if (event && strcmp(event, "prefill_chunk") == 0) {
        if (g->prefill_last_current >= 0 && current > g->prefill_last_current) {
            g->prefill_chunks_done++;
        }
        if (current > g->prefill_last_current) g->prefill_last_current = current;
        g->prefill_total = total;
    }
    server_progress_cb(&g->progress, event, current, total);
}



/* One prefill chunk per quantum. Only interrupt when the engine guarantees
 * bit-exact resumption AND enough suffix remains that the resumed sync takes
 * the batched chunk path rather than the single-token tail path (see
 * ds4_session_prefill_quantum_min_suffix). */
static bool gen_prefill_cancel_cb(void *ud) {
    const gen_state *g = ud;
    if (g->prefill_min_suffix == 0) return false;
    if (g->prefill_chunks_done < 1) return false;
    if (g->prefill_last_current < 0 || g->prefill_total <= g->prefill_last_current) return false;
    return (uint32_t)(g->prefill_total - g->prefill_last_current) >= g->prefill_min_suffix;
}



/* Shared failure epilogue for both prefill phases (the old duplicated blocks
 * after each ds4_session_sync failure). Token vectors and the disk path are
 * freed centrally by gen_state_free. */
static void gen_prefill_fail(server *s, session_slot *sl) {
    gen_state *g = sl->gen;
    ds4_session_set_cancel(sl->sess, NULL, NULL);
    ds4_session_set_progress(sl->sess, NULL, NULL);
    ds4_session_set_display_progress(sl->sess, NULL, NULL);
    kv_cache_tracker_bind(s, sl);
    kv_cache_restore_suppressed_continued(&s->kv, g->suppressed_continued_last,
                                          g->cold_store_len);
    kv_cache_tracker_flush(s, sl);
    kv_cache_discard_failed_disk_entry(s, sl, g->disk_cache_path);
    trace_event(s, g->trace_id, "prefill failed: %s", g->err);
    send_prefill_failure_response(s, g->j, &g->progress, g->ctx_span,
                                  g->req_flags, g->err);
    g->phase = GEN_DONE;
}



/* Resolve the prompt against every cache layer and decide the prefill plan.
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
static void gen_begin(server *s, session_slot *sl) {
    gen_state *g = sl->gen;
    job *j = g->j;
    const int old_pos = ds4_session_pos(sl->sess);
    const int common = ds4_session_common_prefix(sl->sess, &j->req.prompt);
    trace_cache_diag cache_diag = {0};
    trace_cache_capture(&cache_diag, ds4_session_tokens(sl->sess),
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
    int cached = responses_live_visible_prefix_prompt(s, sl, &j->req, old_pos,
                                                      &effective_prompt);
    const char *cache_source = cached > 0 ? "responses-visible" : "none";
    if (cached > 0) {
        responses_live_match = "visible-prefix";
        if (responses_live_matches_request(s, sl, &j->req.responses_live_call_ids,
                                           old_pos))
        {
            responses_live_match_ids = j->req.responses_live_call_ids.len;
        }
    }
    if (cached == 0) {
        cached = responses_live_continuation_prompt(s, sl, &j->req, old_pos,
                                                    &effective_prompt,
                                                    &responses_live_match_ids);
        cache_source = cached > 0 ? "responses-tool-output" : "none";
        if (cached > 0) responses_live_match = "tool-output-ids";
    }
    if (cached > 0) {
        responses_live_continuation = true;
        prompt_for_sync = &effective_prompt;
    } else {
        cached = anthropic_live_continuation_prompt(s, sl, &j->req, old_pos,
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
        g->phase = GEN_DONE;
        return;
    } else if (cached == 0 && j->req.api == API_ANTHROPIC &&
               j->req.anthropic_requires_live_tool_state)
    {
        ds4_tokens_free(&effective_prompt);
        http_error(j->fd, s->enable_cors, 409,
                   "Anthropic continuation state is not available; retry by replaying the full messages history");
        g->phase = GEN_DONE;
        return;
    } else if (cached == 0) {
        cached = common == old_pos && j->req.prompt.len >= old_pos ? common : 0;
        cache_source = cached > 0 ? "memory-token" : "none";
    }
    if (cached == 0) {
        int thinking_cached =
            thinking_live_visible_prefix_prompt(s, sl, &j->req, old_pos,
                                                &effective_prompt);
        if (thinking_cached > 0) {
            cached = thinking_cached;
            cache_source = "thinking-visible";
            thinking_live_continuation = true;
            prompt_for_sync = &effective_prompt;
        }
    }
    int disk_cached = 0;
    uint8_t disk_cache_ext_flags = 0;
    if (cached == 0) {
        int text_cached = live_text_prefix_prompt(s, sl, &j->req, &effective_prompt);
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
    if (cached == 0) sl->continued_last_store_tokens = 0;
    if (s->kv.enabled && cached == 0 && old_pos >= s->kv.opt.min_tokens) {
        /* Loading a disk snapshot replaces the live GPU session.  Persist the
         * current checkpoint first, otherwise a cache hit for an older prefix
         * would silently discard the newer conversation state. */
        kv_cache_store_current(s, sl, "evict");
    }
    if (cached == 0) {
        disk_cached = kv_cache_try_load(s, sl, &j->req, &effective_prompt,
                                        &g->disk_cache_path,
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

    /* Prometheus /metrics: prompt-throughput + prefix-cache-hit counters. */
    pthread_mutex_lock(&s->mu);
    s->m_prompt_tokens += (uint64_t)(prompt_tokens > 0 ? prompt_tokens : 0);
    s->m_prefix_queries += (uint64_t)(prompt_tokens > 0 ? prompt_tokens : 0);
    s->m_prefix_hits += (uint64_t)(cached > 0 ? cached : 0);
    pthread_mutex_unlock(&s->mu);

    g->prompt_tokens = prompt_tokens;
    g->t0 = server_now_sec();
    g->trace_id = trace_begin(s, j, cached, prompt_tokens, &cache_diag,
                              cache_source, disk_cached, g->disk_cache_path);
    request_ctx_span(g->ctx_span, sizeof(g->ctx_span), cached, prompt_tokens);
    g->progress = (server_prefill_progress){
        .srv = s,
        .slot = sl,
        .kind = j->req.kind,
        .prompt_tokens = prompt_tokens,
        .cached_tokens = cached,
        .has_tools = j->req.has_tools,
        .responses_protocol = responses_protocol,
        .t0 = g->t0,
        .fd = j->fd,
        .stream = j->req.stream,
        .enable_cors = s->enable_cors,
    };
    snprintf(g->progress.ctx, sizeof(g->progress.ctx), "%s", g->ctx_span);
    log_flags(g->req_flags, sizeof(g->req_flags), responses_protocol,
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
        trace_event(s, g->trace_id,
                    "responses replay missing reasoning state; continuing from visible history source=%s cached=%d",
                    cache_source, cached);
    }
    server_log(DS4_LOG_PREFILL,
               "ds4-server: %s ctx=%s%s%s prompt start",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               g->ctx_span,
               g->req_flags[0] ? " " : "",
               g->req_flags);
    ds4_session_set_progress(sl->sess, gen_prefill_progress_cb, g);
    ds4_session_set_display_progress(sl->sess, server_progress_cb, &g->progress);

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
    g->cold_store_len = cold_store_len;
    g->suppressed_continued_last = -1;
    if (cold_store_len >= s->kv.opt.min_tokens) {
        /* A cold checkpoint can land exactly on the continued-checkpoint
         * frontier.  The prefill progress callback would then write the same
         * prefix as "continued" while we are intentionally stopping there to
         * write it as "cold".  Mark the frontier as already handled before the
         * sync reaches it; if the cold write fails, restore the old schedule so
         * a later continued write can still try. */
        kv_cache_tracker_bind(s, sl);
        g->suppressed_continued_last =
            kv_cache_suppress_continued_store(&s->kv, cold_store_len);
        kv_cache_tracker_flush(s, sl);
    }

    /* Transfer prompt ownership into the slot state; the prefill phases run in
     * later quanta. */
    g->effective_prompt = effective_prompt;
    g->prompt_for_sync = prompt_for_sync == &effective_prompt ?
                         &g->effective_prompt : prompt_for_sync;
    g->responses_protocol = responses_protocol;
    g->responses_live_continuation = responses_live_continuation;
    g->anthropic_live_continuation = anthropic_live_continuation;
    g->thinking_live_continuation = thinking_live_continuation;

    /* Prefill quantum policy: interrupt the engine's chunk loop only when
     * resumption is bit-exact for this session (see gen_prefill_cancel_cb). */
    g->prefill_min_suffix = ds4_session_prefill_quantum_min_suffix(sl->sess);
    ds4_session_set_cancel(sl->sess, gen_prefill_cancel_cb, g);

    if (s->kv.enabled &&
        g->cold_store_len >= s->kv.opt.min_tokens &&
        g->cold_store_len < g->prompt_for_sync->len)
    {
        tokens_copy_prefix(&g->cold_prefix, g->prompt_for_sync, g->cold_store_len);
        g->phase = GEN_PREFILL_COLD;
    } else {
        g->phase = GEN_PREFILL_MAIN;
    }
}



/* One prefill quantum: (re-)issue the sync toward the phase's target; the
 * cancel callback stops it after one completed chunk and the checkpoint
 * carries the progress to the next quantum. */
static void gen_step_prefill(server *s, session_slot *sl) {
    gen_state *g = sl->gen;
    const bool cold = g->phase == GEN_PREFILL_COLD;
    const ds4_tokens *target = cold ? &g->cold_prefix : g->prompt_for_sync;

    g->prefill_chunks_done = 0;
    g->prefill_last_current = -1;
    g->prefill_total = 0;
    const int rc = ds4_session_sync(sl->sess, target, g->err, sizeof(g->err));
    if (rc == DS4_SESSION_SYNC_INTERRUPTED) {
        if (g->prefill_chunks_done > 0) return; /* voluntary yield; resume next quantum */
        /* Interrupted without progress cannot be our cancel callback; fail
         * rather than risk a live-lock re-issuing the same sync forever. */
        gen_prefill_fail(s, sl);
        return;
    }
    if (rc != 0) {
        gen_prefill_fail(s, sl);
        return;
    }

    if (cold) {
        kv_cache_tracker_bind(s, sl);
        if (kv_cache_store_live_prefix(s, sl, g->prompt_for_sync, g->cold_store_len, "cold")) {
            kv_cache_note_store(&s->kv, g->cold_store_len);
            g->suppressed_continued_last = -1;
        } else {
            kv_cache_restore_suppressed_continued(&s->kv, g->suppressed_continued_last,
                                                  g->cold_store_len);
            g->suppressed_continued_last = -1;
        }
        kv_cache_tracker_flush(s, sl);
        ds4_tokens_free(&g->cold_prefix);
        g->phase = GEN_PREFILL_MAIN;
        return; /* the cold store is a quantum boundary of its own */
    }

    ds4_session_set_cancel(sl->sess, NULL, NULL);
    gen_stream_begin(s, sl);
}

/* Runs once, in the same quantum that completed the main prefill: clear stale
 * live bindings, persist checkpoints, emit response identity, and start the
 * protocol stream projections that persist across all decode quanta. */
static void gen_stream_begin(server *s, session_slot *sl) {
    gen_state *g = sl->gen;
    job *j = g->j;
    free(g->disk_cache_path);
    g->disk_cache_path = NULL;
    /* Once a non-live request wins, old protocol live bindings are stale. Keep
     * a binding only when this request explicitly continued from it. */
    if (!g->responses_live_continuation) responses_live_clear(s, sl);
    if (!g->anthropic_live_continuation) anthropic_live_clear(s, sl);
    if (!g->thinking_live_continuation) thinking_live_clear(s, sl);
    ds4_session_set_progress(sl->sess, NULL, NULL);
    ds4_session_set_display_progress(sl->sess, NULL, NULL);
    kv_cache_maybe_store_continued(s, sl);
    server_log(DS4_LOG_PREFILL,
               "ds4-server: %s ctx=%s%s%s prompt done %.3fs",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               g->ctx_span,
               g->req_flags[0] ? " " : "",
               g->req_flags,
               server_now_sec() - g->t0);
    if (g->cold_store_len == g->prompt_for_sync->len) {
        kv_cache_tracker_bind(s, sl);
        if (kv_cache_store_live_prefix(s, sl, g->prompt_for_sync, g->cold_store_len, "cold")) {
            kv_cache_note_store(&s->kv, g->cold_store_len);
            g->suppressed_continued_last = -1;
        } else {
            kv_cache_restore_suppressed_continued(&s->kv, g->suppressed_continued_last,
                                                  g->cold_store_len);
        }
        kv_cache_tracker_flush(s, sl);
    }
    snprintf(g->id, sizeof(g->id), "%s-%llu",
             j->req.kind == REQ_CHAT ? "chatcmpl" : "cmpl",
             (unsigned long long)++s->seq);

    g->structured_stream = request_uses_structured_stream(&j->req);
    g->openai_live_chat = request_uses_openai_live_stream(&j->req);
    g->responses_live_chat = request_uses_responses_live_stream(&j->req);
    g->responses_created_at = (long)time(NULL);
    if (j->req.stream) {
        if (g->progress.stream_failed) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s%s%s stream closed during prefill",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags);
            g->phase = GEN_DONE;
            return;
        }
        /* The prefill progress callback may have already sent the SSE headers
         * to keep the connection alive during a long prefill. Only emit them
         * here when prefill never fired (e.g. fully cached prompt). */
        if (!g->progress.headers_sent && !sse_headers(j->fd, s->enable_cors)) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s%s%s sse headers failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags);
            g->phase = GEN_DONE;
            return;
        }
        g->progress.headers_sent = true;
        if (j->req.api == API_ANTHROPIC &&
            !anthropic_sse_start_live(j->fd, &j->req, g->id,
                                      g->prompt_tokens, &g->anthropic_live)) {
            server_log(DS4_LOG_GENERATION, "ds4-server: chat ctx=%s anthropic stream start failed", g->ctx_span);
            g->phase = GEN_DONE;
            return;
        }
        if (j->req.api == API_OPENAI && j->req.kind == REQ_CHAT &&
            !sse_chunk(j->fd, &j->req, g->id, NULL, NULL)) {
            server_log(DS4_LOG_GENERATION, "ds4-server: chat ctx=%s openai role chunk failed", g->ctx_span);
            g->phase = GEN_DONE;
            return;
        }
        if (g->openai_live_chat) openai_stream_start(&j->req, &g->openai_live);
        if (g->responses_live_chat) {
            responses_stream_init(&j->req, &g->responses_live);
            g->responses_live.active = true;
            if (!responses_sse_created(j->fd, &j->req, &g->responses_live, g->responses_created_at)) {
                server_log(DS4_LOG_GENERATION,
                           "ds4-server: chat ctx=%s%s%s responses created event failed",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags);
                g->phase = GEN_DONE;
                return;
            }
        }
    }

    g->dsml_recovery_attempted = false;
    g->rng = j->req.seed ? j->req.seed :
        (((uint64_t)time(NULL) << 32) ^ ((uint64_t)s->seq << 1) ^ (uint64_t)(uintptr_t)j);
    g->phase = GEN_DECODE_INIT;
}



/* (Re)initialize a decode attempt: the body of the old decode_again label.
 * Runs both for a fresh request and after a tool-error recovery appended a
 * model-visible correction to the live session. */
static void gen_decode_init(server *s, session_slot *sl) {
    gen_state *g = sl->gen;
    job *j = g->j;
    buf_free(&g->text);
    g->plain_stream_pos = 0;
    g->stop_scan_from = 0;
    g->finish = "length";
    g->completion = 0;
    g->max_tokens = j->req.max_tokens;
    int room = ds4_session_ctx(sl->sess) - ds4_session_pos(sl->sess);
    g->saw_tool_start = false;
    g->saw_tool_end = false;
    g->saw_orphan_tool_end = false;
    g->tool_scan_from = 0;
    g->next_tool_progress = 128;
    g->next_decode_log = 50;
    if (g->max_tokens < 0) g->max_tokens = 0;
    if (g->max_tokens > room) g->max_tokens = room;
    trace_event(s, g->trace_id, "prefill done; decode_max=%d ctx_room=%d", g->max_tokens, room);
    g->decode_t0 = server_now_sec();
    g->last_decode_log_t = g->decode_t0;
    g->last_decode_log_completion = 0;
    g->thinking = thinking_state_from_prompt(&j->req);
    g->thinking_gates_tool_markers = ds4_think_mode_enabled(j->req.think_mode);
    g->tool_scan_waiting_for_think_close =
        g->thinking_gates_tool_markers && g->thinking.inside;
    g->think_recovery_scan_from = 0;
    g->think_tool_recovery_enabled =
        getenv("DS4_SERVER_DISABLE_THINK_TOOL_RECOVERY") == NULL;
    g->dspark_spec_enabled = getenv("DS4_DSPARK_DISABLE") == NULL;
    dsml_decode_tracker_init(&g->dsml_tracker);

    /* tool_choice="required": the prompt was prefilled into an open DSML
     * tool_calls block (thinking skipped: prompt ends "</think>\n\n<tool_calls>").
     * Seed the output with that exact prefix — including the closing </think> so
     * the (thinking-mode) parser sees reasoning end and a complete tool block —
     * and prime the trackers to "inside tool call"; the model now generates only
     * the invoke body. */
    if (j->req.kind == REQ_CHAT && j->req.force_tool_call) {
        request_forced_tool_seed(&j->req, &g->text);
        g->saw_tool_start = true;
        g->tool_scan_waiting_for_think_close = false;
        dsml_decode_tracker_update(&g->dsml_tracker, g->text.ptr, g->text.len);
        g->tool_scan_from = g->text.len;
        g->plain_stream_pos = g->text.len;
    }
    g->phase = GEN_DECODE;
}



/* One decode quantum: run the sampling loop for at most
 * DS4_SERVER_DECODE_QUANTUM_TOKENS generated tokens, then yield with all loop
 * state parked in gen_state. The session is untouched between quanta, so
 * resuming is exactly the next iteration of the old run-to-completion loop. */
static void gen_step_decode(server *s, session_slot *sl) {
    gen_state *g = sl->gen;
    job *j = g->j;
    const int quantum_start = g->completion;
    bool stop_decode = false;

    while (!g_stop_requested && g->completion < g->max_tokens &&
           ds4_session_pos(sl->sess) < ds4_session_ctx(sl->sess)) {
        if (g->completion - quantum_start >= DS4_SERVER_DECODE_QUANTUM_TOKENS) {
            sl->tokens_emitted += (uint64_t)(g->completion - quantum_start);
            return; /* quantum exhausted; phase stays GEN_DECODE */
        }
        dsml_decode_state dsml_state = j->req.kind == REQ_CHAT && j->req.has_tools ?
            g->dsml_tracker.decode : DSML_DECODE_OUTSIDE;
        const bool in_tool_call = dsml_decode_state_is_tool(dsml_state);
        if (!(j->req.kind == REQ_CHAT && j->req.has_tools && (g->saw_tool_start || in_tool_call))) {
            kv_cache_maybe_store_continued(s, sl);
        }
        float temperature = j->req.temperature;
        int top_k = j->req.top_k;
        float top_p = j->req.top_p;
        float min_p = j->req.min_p;
        /* Thinking mode normally forces sampling defaults for quality, but an
         * EXPLICIT temperature==0 (default is 1.0) means the caller wants greedy
         * decode — honor it so DSpark speculative decode (greedy-only) can engage. */
        if (ds4_think_mode_enabled(j->req.think_mode) && j->req.temperature > 0.0f) {
            temperature = DS4_DEFAULT_TEMPERATURE;
            top_k = 0;
            top_p = DS4_DEFAULT_TOP_P;
            min_p = DS4_DEFAULT_MIN_P;
        }
        if (in_tool_call && !dsml_decode_state_uses_payload_sampling(dsml_state)) {
            temperature = 0.0f;
        }
        int token;
        int toks[17];
        int ntok = 0;
        if (ds4_engine_has_dspark(s->engine) && g->dspark_spec_enabled) {
            /* the speculative block owns sampling (exact sampled acceptance at
             * any temperature; greedy degenerates to the argmax rule) */
            ntok = ds4_session_generate_speculative(sl->sess,
                                                    temperature, top_k, top_p, min_p,
                                                    &g->rng,
                                                    g->max_tokens - g->completion,
                                                    ds4_token_eos(s->engine),
                                                    toks,
                                                    (int)(sizeof(toks) / sizeof(toks[0])),
                                                    g->err,
                                                    sizeof(g->err));
            if (ntok < 0) {
                g->finish = "error";
                break;
            }
        } else {
            token = ds4_session_sample(sl->sess, temperature, top_k, top_p, min_p, &g->rng);
            if (token == ds4_token_eos(s->engine)) {
                g->finish = "stop";
                break;
            }
            if (ds4_session_eval(sl->sess, token, g->err, sizeof(g->err)) != 0) {
                g->finish = "error";
                break;
            }
            toks[0] = token;
            ntok = 1;
        }

        for (int ti = 0; ti < ntok && g->completion < g->max_tokens; ti++) {
            token = toks[ti];
            if (token == ds4_token_eos(s->engine)) {
                g->finish = "stop";
                stop_decode = true;
                break;
            }

            size_t piece_len = 0;
            char *piece = ds4_token_text(s->engine, token, &piece_len);
            g->completion++;

            trace_piece(s, g->trace_id, piece, piece_len);
            buf_append(&g->text, piece, piece_len);
            thinking_state_feed(&g->thinking, piece, piece_len);
            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                dsml_decode_tracker_update(&g->dsml_tracker, g->text.ptr, g->text.len);
            }

            size_t stop_pos = 0, stop_len = 0;
            bool hit_stop = stop_list_find_from(&j->req.stops, g->text.ptr,
                                                g->stop_scan_from,
                                                &stop_pos, &stop_len);
            size_t stream_len = hit_stop ?
                stop_pos : stop_list_stream_safe_len(&j->req.stops, g->text.len);
            if (stream_len > g->text.len) stream_len = g->text.len;
            stream_len = utf8_stream_safe_len(g->text.ptr, g->plain_stream_pos,
                                              stream_len, hit_stop);
            if (!hit_stop && j->req.stops.max_len > 1) {
                const size_t hold = j->req.stops.max_len - 1;
                g->stop_scan_from = g->text.len > hold ? g->text.len - hold : 0;
            }

            if (j->req.stream && !g->structured_stream && stream_len > g->plain_stream_pos) {
                char *delta = xstrndup(g->text.ptr + g->plain_stream_pos, stream_len - g->plain_stream_pos);
                bool ok = sse_chunk(j->fd, &j->req, g->id, delta, NULL);
                free(delta);
                if (!ok) {
                    g->finish = "error";
                    snprintf(g->err, sizeof(g->err), "client stream write failed");
                    free(piece);
                    stop_decode = true;
                    break;
                }
                g->plain_stream_pos = stream_len;
            }
            if (j->req.stream && j->req.api == API_ANTHROPIC &&
                !anthropic_sse_stream_update(j->fd, s, &j->req, g->id,
                                             &g->anthropic_live, g->text.ptr, stream_len,
                                             false)) {
                g->finish = "error";
                snprintf(g->err, sizeof(g->err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            if (g->openai_live_chat &&
                !openai_sse_stream_update(j->fd, s, &j->req, g->id,
                                          &g->openai_live, g->text.ptr, stream_len,
                                          false)) {
                g->finish = "error";
                snprintf(g->err, sizeof(g->err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            if (g->responses_live_chat &&
                !responses_sse_stream_update(j->fd, &j->req,
                                             &g->responses_live, g->text.ptr, stream_len,
                                             false)) {
                g->finish = "error";
                snprintf(g->err, sizeof(g->err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            free(piece);

            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                if (g->thinking_gates_tool_markers && g->thinking.inside) {
                    /* A DSML block inside reasoning is not executable.  This is
                     * the live guard: do not let a quoted or mistaken marker in
                     * <think> stop decoding as a real tool call.  A complete
                     * stanza opening, however, almost always means the model
                     * forgot to close its thinking; recover by forcing the
                     * close so the model restarts the call on the executable
                     * side. */
                    const int recovered = g->think_tool_recovery_enabled ?
                        chat_think_tool_recovery(s, sl, &g->text, &g->thinking,
                                                 &g->think_recovery_scan_from,
                                                 &g->completion, g->max_tokens,
                                                 g->err, sizeof(g->err)) : 0;
                    if (recovered < 0) {
                        g->finish = "error";
                        stop_decode = true;
                        break;
                    }
                    if (recovered) {
                        server_log(DS4_LOG_WARNING,
                                   "ds4-server: chat ctx=%s%s%s tool call inside unclosed <think>; "
                                   "forced </think> after %d generated tokens",
                                   g->ctx_span,
                                   g->req_flags[0] ? " " : "",
                                   g->req_flags,
                                   g->completion);
                        trace_event(s, g->trace_id,
                                    "think tool recovery after %d generated tokens",
                                    g->completion);
                        dsml_decode_tracker_update(&g->dsml_tracker, g->text.ptr, g->text.len);
                        g->tool_scan_waiting_for_think_close = true;
                    } else {
                        g->tool_scan_waiting_for_think_close = true;
                        g->tool_scan_from = g->text.len;
                    }
                } else {
                    if (g->tool_scan_waiting_for_think_close) {
                        const char *think_end = find_last_substr(g->text.ptr, "</think>");
                        g->tool_scan_from = think_end ? (size_t)((think_end + 8) - g->text.ptr) : g->text.len;
                        if (g->tool_scan_from > g->text.len) g->tool_scan_from = g->text.len;
                        g->tool_scan_waiting_for_think_close = false;
                    }
                    if (g->tool_scan_from > g->text.len) g->tool_scan_from = g->text.len;
                    const char *tool_scan = g->text.ptr ? g->text.ptr + g->tool_scan_from : "";
                    bool orphan_end = false;
                    bool old_start = g->saw_tool_start;
                    bool old_end = g->saw_tool_end;
                    observe_tool_markers(tool_scan, &g->saw_tool_start, &g->saw_tool_end, &orphan_end);
                    if (orphan_end && !g->saw_orphan_tool_end) {
                        g->saw_orphan_tool_end = true;
                        server_log(DS4_LOG_WARNING,
                                   "ds4-server: chat ctx=%s%s%s ignored orphan tool-call end marker after %d generated tokens",
                                   g->ctx_span,
                                   g->req_flags[0] ? " " : "",
                                   g->req_flags,
                                   g->completion);
                        trace_event(s, g->trace_id,
                                    "ignored orphan tool-call end marker after %d generated tokens",
                                    g->completion);
                    }
                    if (g->saw_tool_start && !old_start) {
                        trace_event(s, g->trace_id, "entered tool-call block after %d generated tokens", g->completion);
                    }
                    if (g->saw_tool_end && !old_end) {
                        trace_event(s, g->trace_id, "closed tool-call block after %d generated tokens", g->completion);
                    }
                    const size_t marker_hold = 80;
                    size_t hold_from = g->text.len > marker_hold ? g->text.len - marker_hold : 0;
                    if (hold_from > g->tool_scan_from) g->tool_scan_from = hold_from;
                    if (s->trace && g->completion >= g->next_tool_progress) {
                        trace_event(s, g->trace_id,
                                    "progress gen=%d dsml_start=%d dsml_end=%d",
                                    g->completion, g->saw_tool_start ? 1 : 0, g->saw_tool_end ? 1 : 0);
                        g->next_tool_progress += 128;
                    }
                }
            }

            if (g->completion >= g->next_decode_log) {
                log_decode_progress(j->req.kind, g->prompt_tokens, g->completion,
                                    g->responses_protocol,
                                    j->req.has_tools,
                                    g->thinking.inside,
                                    g->saw_tool_start,
                                    g->saw_tool_end,
                                    g->decode_t0,
                                    &g->last_decode_log_t,
                                    &g->last_decode_log_completion);
                g->next_decode_log += 50;
            }

            if (hit_stop) {
                (void)stop_len;
                g->finish = "stop";
                g->text.len = stop_pos;
                g->text.ptr[g->text.len] = '\0';
                ds4_session_invalidate(sl->sess);
                stop_decode = true;
                break;
            }

            if (j->req.kind == REQ_CHAT && j->req.has_tools && g->saw_tool_end) {
                g->finish = "tool_calls";
                stop_decode = true;
                break;
            }
        }
        if (stop_decode) break;
    }

    sl->tokens_emitted += (uint64_t)(g->completion - quantum_start);
    g->phase = GEN_FINISH;
}



/* Post-decode epilogue: tool repair/recovery, final parse, protocol live
 * state, checkpoints, the final response, and logging. Recovery paths loop
 * back to GEN_DECODE_INIT (the old goto decode_again). */
static void gen_step_finish(server *s, session_slot *sl) {
    gen_state *g = sl->gen;
    job *j = g->j;

    if (g_stop_requested && strcmp(g->finish, "error") != 0) {
        g->finish = "error";
        snprintf(g->err, sizeof(g->err), "shutdown requested");
    }

    if (j->req.kind == REQ_CHAT && j->req.has_tools &&
        g->saw_tool_start && !g->saw_tool_end && strcmp(g->finish, "error") != 0)
    {
        /* Deterministically complete a simple truncation.  Anything more than
         * missing closing tags stays model-owned: for non-streaming requests,
         * append a tool error plus prompt reminder to the live session and let
         * the model issue a fresh call. */
        bool completed_truncation = false;
        buf repaired = {0};
        if (try_repair_dsml(g->text.ptr, g->text.len, &repaired)) {
            /* Parse repaired text to verify it produces valid tool calls */
            tool_calls test_calls = {0};
            char *test_content = NULL;
            char *test_reasoning = NULL;
            bool repair_ok = parse_generated_message_ex(repaired.ptr, false, &test_content, &test_reasoning, &test_calls);
            free(test_content);
            free(test_reasoning);
            if (repair_ok && test_calls.len > 0) {
                /* Repair succeeded - replace text with repaired version */
                free(g->text.ptr);
                g->text.ptr = buf_take(&repaired);
                g->text.len = strlen(g->text.ptr);
                g->text.cap = g->text.len ? g->text.len + 1 : 0;
                g->saw_tool_end = true;
                completed_truncation = true;
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s repaired unterminated tool call (%d calls recovered)",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags,
                           test_calls.len);
                trace_event(s, g->trace_id, "repaired unterminated tool call (%d calls recovered)", test_calls.len);
            }
            tool_calls_free(&test_calls);
        }
        if (!completed_truncation) {
            if (!j->req.stream && !g->dsml_recovery_attempted) {
                int recovery_tokens = 0;
                char recovery_err[160] = {0};
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s unterminated tool call; continuing with model-visible tool error",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags);
                trace_event(s, g->trace_id,
                            "unterminated tool call; continuing with model-visible tool error");
                if (continue_after_invalid_dsml(s, sl, &j->req, &g->thinking,
                                                "unterminated tool call",
                                                &recovery_tokens,
                                                recovery_err,
                                                sizeof(recovery_err)))
                {
                    g->dsml_recovery_attempted = true;
                    server_log(DS4_LOG_GENERATION,
                               "ds4-server: chat ctx=%s%s%s tool-error continuation appended %d tokens",
                               g->ctx_span,
                               g->req_flags[0] ? " " : "",
                               g->req_flags,
                               recovery_tokens);
                    trace_event(s, g->trace_id,
                                "tool-error continuation appended %d tokens",
                                recovery_tokens);
                    buf_free(&repaired);
                    buf_free(&g->text);
                    g->phase = GEN_DECODE_INIT; /* the old goto decode_again */
                    return;
                }
                g->finish = "error";
                snprintf(g->err, sizeof(g->err), "invalid tool call recovery failed: %s",
                         recovery_err[0] ? recovery_err : "unknown error");
            } else {
                g->finish = "error";
                snprintf(g->err, sizeof(g->err), "unterminated tool call");
            }
        }
        buf_free(&repaired);
    }

    if (g->completion > g->last_decode_log_completion) {
        log_decode_progress(j->req.kind, g->prompt_tokens, g->completion,
                            g->responses_protocol,
                            j->req.has_tools,
                            g->thinking.inside,
                            g->saw_tool_start,
                            g->saw_tool_end,
                            g->decode_t0,
                            &g->last_decode_log_t,
                            &g->last_decode_log_completion);
    }

    if (j->req.stream && !g->structured_stream && g->text.len > g->plain_stream_pos) {
        char *tail = xstrndup(g->text.ptr + g->plain_stream_pos, g->text.len - g->plain_stream_pos);
        if (!sse_chunk(j->fd, &j->req, g->id, tail, NULL)) g->finish = "error";
        free(tail);
    }

    tool_calls parsed_calls = {0};
    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    const char *final_finish = g->finish;
    bool recovered_tool_parse_failure = false;
    if (j->req.kind == REQ_CHAT) {
        bool parsed_ok = parse_generated_message_for_response(
            g->text.ptr ? g->text.ptr : "",
            j->req.has_tools,
            g->saw_tool_start,
            ds4_think_mode_enabled(j->req.think_mode),
            &final_finish,
            g->err,
            sizeof(g->err),
            &parsed_content,
            &parsed_reasoning,
            &parsed_calls,
            &recovered_tool_parse_failure);
        if (!parsed_ok && recovered_tool_parse_failure && j->req.has_tools && g->saw_tool_start) {
            /* parse_generated_message failed even though DSML was present.
             * Semantic repair is intentionally avoided: if the parser cannot
             * execute the block, feed the model a tool error and the protocol
             * reminder so it owns the corrected next action. */
            if (!j->req.stream && !g->dsml_recovery_attempted) {
                int recovery_tokens = 0;
                char recovery_err[160] = {0};
                const char *detail = g->err[0] ? g->err : "invalid tool call";
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s invalid tool call; continuing with model-visible tool error",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags);
                trace_event(s, g->trace_id,
                            "invalid tool call; continuing with model-visible tool error");
                if (continue_after_invalid_dsml(s, sl, &j->req, &g->thinking,
                                                detail,
                                                &recovery_tokens,
                                                recovery_err,
                                                sizeof(recovery_err)))
                {
                    g->dsml_recovery_attempted = true;
                    server_log(DS4_LOG_GENERATION,
                               "ds4-server: chat ctx=%s%s%s tool-error continuation appended %d tokens",
                               g->ctx_span,
                               g->req_flags[0] ? " " : "",
                               g->req_flags,
                               recovery_tokens);
                    trace_event(s, g->trace_id,
                                "tool-error continuation appended %d tokens",
                                recovery_tokens);
                    free(parsed_content);
                    free(parsed_reasoning);
                    tool_calls_free(&parsed_calls);
                    buf_free(&g->text);
                    g->phase = GEN_DECODE_INIT; /* the old goto decode_again */
                    return;
                }
                final_finish = "error";
                snprintf(g->err, sizeof(g->err), "invalid tool call recovery failed: %s",
                         recovery_err[0] ? recovery_err : "unknown error");
            }
            if (!parsed_ok) {
                /* Print raw DSML snippet for debugging */
                size_t dsml_snippet_len = 0;
                const char *dsml_start = NULL;
                const char *p;
                for (p = g->text.ptr; p && (size_t)(p - g->text.ptr) < g->text.len - 20; p++) {
                    if ((strncmp(p, DS4_TOOL_CALLS_START, strlen(DS4_TOOL_CALLS_START)) == 0) ||
                        (strncmp(p, DS4_TOOL_CALLS_START_SHORT, strlen(DS4_TOOL_CALLS_START_SHORT)) == 0) ||
                        (strncmp(p, "<tool_calls>", 12) == 0)) {
                        dsml_start = p;
                        break;
                    }
                }
                if (dsml_start) {
                    dsml_snippet_len = g->text.len - (dsml_start - g->text.ptr);
                    if (dsml_snippet_len > 500) dsml_snippet_len = 500;
                }
                /* Also log a snippet of the full text to see what the model output */
                size_t text_snippet_len = g->text.len > 300 ? 300 : g->text.len;
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s invalid tool call returned as assistant text finish=%s [text_len=%zu saw_start=%d saw_end=%d text_snippet: %.*s]",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags,
                           final_finish,
                           g->text.len,
                           g->saw_tool_start,
                           g->saw_tool_end,
                           (int)text_snippet_len,
                           g->text.ptr ? g->text.ptr : "(null)");
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s invalid tool call dsml_snippet: %.*s",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags,
                           (int)dsml_snippet_len,
                           dsml_start ? dsml_start : "(none)");
                trace_event(s, g->trace_id,
                            "invalid tool call returned as assistant text finish=%s",
                            final_finish);
            }
        }
        if (parsed_calls.len) {
            if (g->openai_live_chat) apply_openai_stream_tool_ids(&parsed_calls, &g->openai_live);
            if (j->req.api == API_ANTHROPIC && j->req.stream)
                apply_anthropic_stream_tool_ids(&parsed_calls, &g->anthropic_live);
            assign_tool_call_ids(s, &parsed_calls, j->req.api);
            tool_memory_remember(s, &parsed_calls);
            final_finish = "tool_calls";
        } else if (j->req.api == API_RESPONSES) {
            responses_live_clear(s, sl);
        }
    }
    log_tool_calls_summary(g->ctx_span, &parsed_calls,
                           g->responses_protocol);

    trace_finish(s, g->trace_id, &j->req, final_finish, g->completion,
                 g->saw_tool_start, g->saw_tool_end,
                 parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                 parsed_reasoning, &parsed_calls, server_now_sec() - g->t0);

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
            responses_live_remember(s, sl, visible.ptr ? visible.ptr : "",
                                    parsed_calls.len ? &parsed_calls : NULL);
            buf_free(&visible);
            free(visible_suffix);
        } else {
            responses_live_clear(s, sl);
        }
    }
    if (j->req.api == API_ANTHROPIC) {
        if (parsed_calls.len && strcmp(final_finish, "error") &&
            strcmp(final_finish, "length"))
        {
            anthropic_live_remember(s, sl, &parsed_calls);
        } else {
            anthropic_live_clear(s, sl);
        }
    }

    if (j->req.kind == REQ_CHAT && parsed_calls.len &&
        j->req.api != API_RESPONSES &&
        ds4_think_mode_enabled(j->req.think_mode) &&
        !j->req.force_tool_call)
    {
        /* Tool call with thinking on: the reasoning is in the live KV but the
         * client replays the turn stripped, so exact-DSML replay and
         * canonicalization both fail (the latter can only re-prefill). Remember
         * the stripped bytes as a key and keep the live tokens — the next
         * request continues from live KV with no rebuild. */
        remember_tool_thinking_checkpoint(s, sl, j, g->ctx_span, g->trace_id,
                                          parsed_content ? parsed_content : "",
                                          &parsed_calls);
    } else if (j->req.kind == REQ_CHAT && parsed_calls.len &&
        j->req.api != API_RESPONSES &&
        should_canonicalize_tool_checkpoint(s, &parsed_calls))
    {
        /* Chat/completions has no protocol object that binds the next request
         * to this live KV state.  Canonicalize only the fallback tool-call
         * path where we lack exact sampled DSML replay; when raw DSML is known,
         * replaying those bytes keeps future prompts aligned without rebuilding
         * hidden reasoning.  Responses deliberately skips this path because its
         * previous_response_id contract binds the next turn to live state. */
        canonicalize_tool_checkpoint(s, sl, j, g->ctx_span, g->trace_id,
                                     parsed_content ? parsed_content : "",
                                     parsed_reasoning, &parsed_calls);
        thinking_live_clear(s, sl);
    } else if (parsed_calls.len) {
        thinking_live_clear(s, sl);
    } else if (!parsed_calls.len &&
               should_remember_thinking_checkpoint(&j->req, &g->thinking, final_finish)) {
        remember_thinking_checkpoint(s, sl, j, g->ctx_span, g->trace_id,
                                     parsed_content ? parsed_content : "");
    } else if (!parsed_calls.len) {
        thinking_live_clear(s, sl);
    }

    if (j->req.stream) {
        bool response_ok = true;
        if (j->req.api == API_ANTHROPIC) {
            response_ok = anthropic_sse_finish_live(j->fd, s, &j->req, g->id, &g->anthropic_live,
                                                    g->text.ptr ? g->text.ptr : "", g->text.len,
                                                    &parsed_calls, final_finish, g->completion);
        } else if (g->openai_live_chat) {
            response_ok = openai_sse_finish_live(j->fd, s, &j->req, g->id, &g->openai_live,
                                                 g->text.ptr ? g->text.ptr : "", g->text.len,
                                                 &parsed_calls, final_finish,
                                                 g->prompt_tokens, g->completion);
        } else if (g->responses_live_chat) {
            /* If parse recovered a malformed tool call back to plain text,
             * pass parsed_content so the streaming tail can be flushed; in
             * the normal path parsed_content is the assistant text we already
             * streamed and the diff is empty. */
            const char *recover =
                recovered_tool_parse_failure ? parsed_content : NULL;
            response_ok = responses_sse_finish_live(j->fd, &j->req, &g->responses_live,
                                                    g->text.ptr ? g->text.ptr : "", g->text.len,
                                                    recover,
                                                    &parsed_calls, final_finish,
                                                    g->prompt_tokens, g->completion,
                                                    g->responses_created_at);
        } else if (g->structured_stream) {
            response_ok = sse_chat_finish(j->fd, &j->req, g->id,
                                          parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                                          parsed_reasoning,
                                          &parsed_calls, final_finish,
                                          g->prompt_tokens, g->completion);
        } else {
            response_ok = sse_chunk(j->fd, &j->req, g->id, NULL, final_finish) &&
                          sse_done(j->fd, &j->req, g->id, g->prompt_tokens, g->completion);
        }
        if (!response_ok) {
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: %s ctx=%s%s%s final stream failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags);
        }
    } else if (j->req.api == API_ANTHROPIC) {
        anthropic_final_response(j->fd, s->enable_cors, &j->req, g->id,
                                 parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 g->prompt_tokens, g->completion);
    } else if (j->req.api == API_RESPONSES) {
        responses_final_response(j->fd, s->enable_cors, &j->req, g->id,
                                 parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 g->prompt_tokens, g->completion);
    } else {
        final_response(j->fd, s->enable_cors, &j->req, g->id,
                       parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                       parsed_reasoning,
                       &parsed_calls, final_finish,
                       g->prompt_tokens, g->completion);
    }
    if (j->req.kind == REQ_CHAT && j->req.has_tools) {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  g->responses_protocol,
                  true,
                  g->thinking.inside,
                  g->saw_tool_start,
                  g->saw_tool_end);
        if (!strcmp(final_finish, "error") && g->err[0]) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: chat ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       g->err,
                       server_now_sec() - g->t0);
        } else {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: chat ctx=%s gen=%d%s%s finish=%s %.3fs",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       server_now_sec() - g->t0);
        }
    } else {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  g->responses_protocol,
                  j->req.has_tools,
                  g->thinking.inside,
                  false,
                  false);
        if (!strcmp(final_finish, "error") && g->err[0]) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       g->err,
                       server_now_sec() - g->t0);
        } else {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s gen=%d%s%s finish=%s %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       server_now_sec() - g->t0);
        }
    }
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&parsed_calls);
    g->phase = GEN_DONE;
}



/* ---- state-machine driver: bind, step, unbind ---- */

static void gen_state_free(server *s, session_slot *sl) {
    gen_state *g = sl->gen;
    if (!g) return;
    (void)s;
    /* Callback safety: no gen_state pointer may remain installed anywhere. */
    ds4_session_set_cancel(sl->sess, NULL, NULL);
    ds4_session_set_progress(sl->sess, NULL, NULL);
    ds4_session_set_display_progress(sl->sess, NULL, NULL);
    anthropic_stream_free(&g->anthropic_live);
    openai_stream_free(&g->openai_live);
    responses_stream_free(&g->responses_live);
    buf_free(&g->text);
    ds4_tokens_free(&g->effective_prompt);
    ds4_tokens_free(&g->cold_prefix);
    free(g->disk_cache_path);
    slot_writer_free(&g->writer);
    free(g);
    sl->gen = NULL;
}



/* Bind a dequeued job to the slot and resolve its prompt (the first quantum). */
static void generate_job_begin(server *s, session_slot *sl, job *j) {
    gen_state *g = server_xmalloc(sizeof(*g));
    memset(g, 0, sizeof(*g));
    g->j = j;
    g->prompt_for_sync = &j->req.prompt;
    g->finish = "length";
    g->suppressed_continued_last = -1;
    sl->gen = g;
    sl->active_job = j;
    sl->state = SLOT_PREFILLING;
    /* All client writes for this job (worker thread only) become non-blocking
     * and deferred; drained in generate_job_end. */
    slot_writer_init(&g->writer, j->fd);
    slot_writer_install(&g->writer);
    gen_begin(s, sl);
}



/* Advance the job by one quantum. */
static void generate_job_step(server *s, session_slot *sl) {
    gen_state *g = sl->gen;
    /* The installed slot writer is worker-thread-local and shared across
     * slots; re-install this slot's writer so send_all() routes through the
     * right deferral queue after another slot (or a fresh bind) was serviced
     * in between. */
    slot_writer_install(&g->writer);
    /* Push any bytes a slow client deferred before spending GPU time. */
    slot_writer_flush(&g->writer);
    switch (g->phase) {
    case GEN_PREFILL_COLD:
    case GEN_PREFILL_MAIN:
        sl->state = SLOT_PREFILLING;
        gen_step_prefill(s, sl);
        break;
    case GEN_DECODE_INIT:
        sl->state = SLOT_DECODING;
        gen_decode_init(s, sl);
        if (g->phase != GEN_DECODE) break;
        /* fall through into the first decode quantum */
        gen_step_decode(s, sl);
        break;
    case GEN_DECODE:
        sl->state = SLOT_DECODING;
        gen_step_decode(s, sl);
        break;
    case GEN_FINISH:
        gen_step_finish(s, sl);
        break;
    case GEN_DONE:
        break;
    }
}



/* Unbind: drain deferred client bytes, free the resumable state. */
static void generate_job_end(server *s, session_slot *sl) {
    if (sl->gen) slot_writer_drain(&sl->gen->writer);
    gen_state_free(s, sl);
    sl->active_job = NULL;
    sl->state = SLOT_IDLE;
    sl->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
}



bool enqueue(server *s, job *j) {
    pthread_mutex_lock(&s->mu);
    if (s->stopping) {
        pthread_mutex_unlock(&s->mu);
        return false;
    }
    if (s->tail) s->tail->next = j; else s->head = j;
    s->tail = j;
    s->n_queued++;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
    return true;
}



/* =========================================================================
 * Increment 3: job→slot binding + round-robin scheduler.
 *
 * Everything below runs on the single GPU worker thread. Client threads only
 * enqueue jobs (enqueue above, under mu) and block on the per-job condvar;
 * they never touch a slot or a ds4_session. Over-capacity requests stay in
 * the FIFO queue until a slot frees (plan Tier 1 §1.4 "queue them until a
 * slot frees"); the queue is bounded by DS4_SERVER_MAX_CLIENTS, since every
 * queued job is one connected client thread.
 * ========================================================================= */

/* Context a request needs from a slot: prompt plus generation budget (plus a
 * small allowance for tool-error recovery injections), capped at the largest
 * (startup) slot so every request can always run on slot 0. */
static int job_needed_ctx(const server *s, const job *j) {
    int64_t need = (int64_t)j->req.prompt.len +
                   (int64_t)(j->req.max_tokens > 0 ? j->req.max_tokens
                                                   : s->default_tokens) +
                   64;
    if (need > s->slots[0].ctx_size) need = s->slots[0].ctx_size;
    if (need < 1) need = 1;
    return (int)need;
}



/* Reconcile a session's admission estimate with the bytes the allocator
 * actually committed (2026-07-13 lockup postmortem: the ledger must track
 * reality, not a formula). Logs the pair, warns loudly on >10% drift — that
 * means gpu_graph_session_bytes (gpu_diag.c) has fallen out of sync with
 * gpu_graph_alloc_raw_cap — and returns the value to commit to the ledger
 * (the actual, unless the engine could not measure one). */
uint64_t server_reconciled_session_cost(int slot_idx, int ctx,
                                        uint64_t est_bytes,
                                        uint64_t actual_bytes) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    server_log(DS4_LOG_DEFAULT,
               "ds4-server: slot %d session cost: est=%.2f GiB actual=%.2f GiB (ctx=%d)",
               slot_idx, (double)est_bytes / gib, (double)actual_bytes / gib, ctx);
    if (actual_bytes == 0) return est_bytes;
    if (est_bytes > 0 &&
        (actual_bytes > est_bytes + est_bytes / 10 ||
         est_bytes > actual_bytes + actual_bytes / 10))
    {
        server_log(DS4_LOG_WARNING,
                   "ds4-server: SESSION COST DRIFT >10%%: est=%.2f GiB vs actual=%.2f GiB "
                   "— gpu_graph_session_bytes is out of sync with gpu_graph_alloc_raw_cap "
                   "(or a deliberately unaccounted allocation is enabled: directional-"
                   "steering dirs are in the measured delta but excluded from the "
                   "estimate — see the exclusion list in gpu_diag.c); "
                   "fix the sizing code (gpu_diag.c) before trusting admission control",
                   (double)est_bytes / gib, (double)actual_bytes / gib);
    }
    return actual_bytes;
}



/* MemAvailable from /proc/meminfo, in bytes (0 on parse failure — the caller
 * fails closed and refuses provisioning). One read per slot-provisioning
 * attempt — never on a token/layer hot path. Coarse by design: driver 610's
 * UVM accounting lags MemAvailable, and under UVM pressure MemTotal itself
 * shrinks, so this is a belt-and-suspenders floor check on top of the
 * ledger, not a precise gauge. */
static uint64_t server_mem_available_bytes(void) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    unsigned long long kib = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
    }
    fclose(fp);
    return (uint64_t)kib * 1024ull;
}



/* Lazily provision a fresh slot under the session admission budget (plan
 * Tier 1 §1.4), pricing the session with the TRUE per-session cost
 * (ds4_engine_session_cost_bytes: full graph + prefill working set + drafter
 * state — the packed-KV estimate under-counted ~10x and hard-locked the
 * machine, 2026-07-13). Returns NULL when the pool is at capacity, admission
 * fails, MemAvailable would drop below the floor, or session creation fails —
 * the job then waits in the queue, exactly like an admission rejection.
 * Since increment 4, an evicted slot leaves a hole (sess == NULL) below
 * n_slots; provisioning reuses the lowest hole before extending the pool, so
 * n_slots stays the high-water published count every reader iterates by
 * (they all skip sess == NULL entries). *refusal reports WHY provisioning
 * failed: the eviction path only acts on refusals eviction can actually
 * relieve (a full pool or a full ledger — never the MemAvailable floor,
 * which freed CUDA memory does not promptly move; see worker_try_bind). */
typedef enum {
    PROVISION_OK = 0,
    PROVISION_REFUSED_POOL_FULL,   /* no free slot entry — eviction helps */
    PROVISION_REFUSED_ADMISSION,   /* ledger full — eviction helps */
    PROVISION_REFUSED_MEM_FLOOR,   /* machine physically tight (incl. an
                                      unreadable /proc/meminfo, fail closed) —
                                      eviction does NOT promptly help */
    PROVISION_REFUSED_CREATE_FAIL, /* allocation failed — eviction unsafe to
                                      chain on (same physical pressure) */
} provision_refusal;

static session_slot *provision_slot(server *s, int ctx,
                                    provision_refusal *refusal) {
    *refusal = PROVISION_OK;
    int idx = -1;
    for (int i = 0; i < DS4_SESSION_POOL_CAP; i++) {
        if (!s->slots[i].sess) { idx = i; break; }
    }
    if (idx < 0) { /* pool at capacity */
        *refusal = PROVISION_REFUSED_POOL_FULL;
        return NULL;
    }
    const uint64_t est = ds4_engine_session_cost_bytes(s->engine, ctx);
    pthread_mutex_lock(&s->mu);
    const uint64_t committed = s->kv_committed_bytes;
    pthread_mutex_unlock(&s->mu);
    /* Retried at every scheduling pass while the head job waits; log only when
     * the rejected shape (or the refusing guard) changes, and re-arm after any
     * successful provisioning (single worker thread). */
    static uint64_t last_rejected_est, last_rejected_committed;
    static int last_rejected_reason; /* 0 none, 1 admission, 2 mem floor */
    if (est == 0 || !server_kv_admits(s->kv_budget_bytes, committed, est)) {
        if (last_rejected_reason != 1 ||
            est != last_rejected_est || committed != last_rejected_committed) {
            last_rejected_reason = 1;
            last_rejected_est = est;
            last_rejected_committed = committed;
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: slot admission rejected: ctx=%d est=%.2f GiB "
                       "committed=%.2f GiB budget=%.2f GiB (job queued)",
                       ctx,
                       (double)est / (1024.0 * 1024.0 * 1024.0),
                       (double)committed / (1024.0 * 1024.0 * 1024.0),
                       (double)s->kv_budget_bytes / (1024.0 * 1024.0 * 1024.0));
        }
        *refusal = PROVISION_REFUSED_ADMISSION;
        return NULL;
    }
    /* Belt and suspenders: whatever the ledger says, do not start a create
     * unless the kernel still reports room for it plus the free floor. A
     * /proc/meminfo parse failure REFUSES provisioning (fail closed): this
     * guard exists precisely for when other accounting is wrong, so it must
     * not silently disarm itself. */
    const uint64_t avail = server_mem_available_bytes();
    if (avail == 0) {
        static bool warned_meminfo_unreadable; /* log once; single worker thread */
        if (!warned_meminfo_unreadable) {
            warned_meminfo_unreadable = true;
            server_log(DS4_LOG_WARNING,
                       "ds4-server: /proc/meminfo MemAvailable unreadable; "
                       "refusing slot provisioning (jobs queue for existing slots)");
        }
        *refusal = PROVISION_REFUSED_MEM_FLOOR;
        return NULL;
    }
    if (avail < est + DS4_SERVER_MEM_FLOOR_BYTES) {
        if (last_rejected_reason != 2 ||
            est != last_rejected_est || committed != last_rejected_committed) {
            last_rejected_reason = 2;
            last_rejected_est = est;
            last_rejected_committed = committed;
            server_log(DS4_LOG_WARNING,
                       "ds4-server: slot provisioning refused: MemAvailable %.2f GiB "
                       "< est %.2f GiB + floor %.2f GiB (ctx=%d; job queued)",
                       (double)avail / (1024.0 * 1024.0 * 1024.0),
                       (double)est / (1024.0 * 1024.0 * 1024.0),
                       (double)DS4_SERVER_MEM_FLOOR_BYTES / (1024.0 * 1024.0 * 1024.0),
                       ctx);
        }
        *refusal = PROVISION_REFUSED_MEM_FLOOR;
        return NULL;
    }
    last_rejected_reason = 0; /* provisioning proceeds: re-arm rejection logs */
    last_rejected_est = 0;
    last_rejected_committed = 0;
    ds4_session *sess = NULL;
    if (ds4_session_create(&sess, s->engine, ctx) != 0) {
        server_log(DS4_LOG_WARNING,
                   "ds4-server: slot session create failed (ctx=%d); job queued",
                   ctx);
        *refusal = PROVISION_REFUSED_CREATE_FAIL;
        return NULL;
    }
    const uint64_t actual =
        server_reconciled_session_cost(idx, ctx, est,
                                       ds4_session_resident_bytes(sess));
    session_slot *sl = &s->slots[idx];
    sl->sess = sess;
    sl->state = SLOT_IDLE;
    sl->ctx_size = ctx;
    sl->est_cost_bytes = actual; /* ledger commits ACTUALS */
    sl->tokens_emitted = 0;
    /* A freshly provisioned (or evicted-and-reused) slot must not be the next
     * LRU victim purely because its last-serviced stamp is stale/zero. */
    sl->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
    sl->continued_last_store_tokens = 0;
    pthread_mutex_lock(&s->mu);
    s->kv_committed_bytes += actual;
    /* publish only after the slot is fully initialized */
    if (idx >= s->n_slots) s->n_slots = idx + 1;
    pthread_mutex_unlock(&s->mu);
    server_log(DS4_LOG_DEFAULT,
               "ds4-server: provisioned slot %d ctx=%d committed(actual)=%.2f GiB "
               "total committed=%.2f GiB / %.2f GiB",
               idx, ctx,
               (double)actual / (1024.0 * 1024.0 * 1024.0),
               (double)s->kv_committed_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)s->kv_budget_bytes / (1024.0 * 1024.0 * 1024.0));
    return sl;
}



/* Context a lazily provisioned slot would be created with for this job: the
 * secondary-slot default, raised to the job's need, capped at slot 0's ctx.
 * Shared by the provisioning path and the eviction could-it-help precheck so
 * they price the same session shape. */
static int provision_ctx_for_job(const server *s, const job *j) {
    int ctx = DS4_SERVER_EXTRA_SLOT_CTX_TOKENS;
    if (ctx > s->slots[0].ctx_size) ctx = s->slots[0].ctx_size;
    const int needed = job_needed_ctx(s, j);
    if (ctx < needed) ctx = needed;
    return ctx;
}



/* Route the job to a slot. Preferences, in order:
 *   1. A live-tool-state continuation binds to the slot that owns its call
 *      ids (waiting for it if busy — running it elsewhere could only 409).
 *      A continuation whose prompt cannot fit its owner slot's context can
 *      never run: it must not run elsewhere (the live tool state exists only
 *      on the owner), and leaving it queued would wedge the FIFO forever
 *      behind an unbindable head — so it is failed explicitly through
 *      *reject_ctx with the same context_length_exceeded client error the
 *      front door sends (http_server.c / request_exceeds_context; the front
 *      door checks against slot 0's ctx and cannot see the owner's smaller
 *      one).
 *   2. Among free slots with enough context, the longest common token prefix
 *      wins, keeping a client's follow-ups on their warm KV.
 *   3. A job with no matching prefix anywhere prefers a fresh lazily
 *      provisioned slot over clobbering another conversation's warm state
 *      (budget permitting); with the pool exhausted it falls back to the
 *      warmest free slot exactly like the single-session server did.
 * Returns NULL when the job must wait for a slot to free — except when
 * *reject_ctx is set nonzero (the owner slot's ctx_size), which means the
 * job can never run and must be failed, not left queued. *waiting_owner is
 * set when the NULL means "the continuation's owner slot is busy": eviction
 * cannot help that job, only the owner finishing can. *clobbers is set when
 * the returned slot would overwrite another conversation's warm KV — the
 * caller may prefer evict(LRU)+provision over that (increment 4). *refusal
 * reports why a fresh provisioning was refused (PROVISION_OK when none was
 * attempted or it succeeded) so the eviction path can act only on refusals
 * eviction relieves. */
static session_slot *choose_slot_for_job(server *s, job *j, int *reject_ctx,
                                         bool *waiting_owner, bool *clobbers,
                                         provision_refusal *refusal) {
    *waiting_owner = false;
    *clobbers = false;
    *refusal = PROVISION_OK;
    session_slot *owner = NULL;
    if (j->req.api == API_RESPONSES && j->req.responses_live_call_ids.len > 0) {
        owner = responses_live_slot_for_ids(s, &j->req.responses_live_call_ids);
    } else if (j->req.api == API_ANTHROPIC &&
               j->req.anthropic_live_call_ids.len > 0) {
        owner = anthropic_live_slot_for_ids(s, &j->req.anthropic_live_call_ids);
    }
    if (owner) {
        if (request_exceeds_context(&j->req, owner->ctx_size)) {
            *reject_ctx = owner->ctx_size;
            return NULL;
        }
        *waiting_owner = owner->active_job != NULL;
        return owner->active_job ? NULL : owner;
    }

    const int needed = job_needed_ctx(s, j);
    session_slot *best = NULL;
    int best_common = -1;
    for (int i = 0; i < s->n_slots; i++) {
        session_slot *sl = &s->slots[i];
        if (sl->active_job || !sl->sess) continue;
        if (sl->ctx_size < needed) continue;
        const int common = ds4_session_common_prefix(sl->sess, &j->req.prompt);
        if (common > best_common) {
            best_common = common;
            best = sl;
        }
    }
    const bool best_clobbers_warm_state =
        best && best_common == 0 && ds4_session_pos(best->sess) > 0;
    if (!best || best_clobbers_warm_state) {
        session_slot *fresh = provision_slot(s, provision_ctx_for_job(s, j),
                                             refusal);
        if (fresh) return fresh;
    }
    *clobbers = best_clobbers_warm_state;
    return best;
}



/* =========================================================================
 * Increment 4: LRU eviction of idle slots.
 *
 * Sessions were previously never freed at runtime: once the admission budget
 * was consumed by idle-but-warm sessions, new conversations either queued
 * forever (no fitting free slot) or silently clobbered the warmest idle slot
 * chosen by scan order. Now, when the queue head cannot be placed cleanly —
 * no fitting free slot, or only a slot whose warm KV belongs to another
 * conversation — AND provisioning was refused by a constraint eviction can
 * actually relieve (a full pool or a full admission ledger), the worker
 * evicts the least-recently-serviced IDLE slot: snapshot its KV to the disk
 * kv cache (the same LRU store used for shutdown/cold checkpoints — plan
 * Tier 1 §1.3 "spill target"), free the session, release its ACTUAL
 * committed bytes from the ledger, and retry placement (the freed budget +
 * slot entry let provisioning succeed). The clobber fallback remains when
 * eviction cannot help: it is an in-place eviction of that one slot
 * (gen_begin disk-stores the displaced state), so correctness is identical —
 * LRU eviction just picks a better victim and keeps warmer conversations
 * alive.
 *
 * MemAvailable-floor refusals deliberately do NOT trigger eviction. Measured
 * on the GB10 (driver 610, 2026-07-14 smoke): freeing two 2.5 GiB sessions
 * moved MemAvailable only 5.98 -> 6.07 GiB — cudaFree'd memory does not
 * promptly return to the kernel's gauge, so post-eviction provisioning kept
 * refusing on the same floor while two warm conversations were lost for
 * nothing (and the server degraded to one effective slot). A floor refusal
 * means the machine is physically tight; the honest response is to queue the
 * head until a slot frees, not to churn snapshots.
 *
 * Restore path: none of it is new. gen_begin's cache resolution already
 * prefers a disk-text snapshot over cold prefill (kv_cache_try_load), and the
 * "evict" snapshot is keyed exactly like a shutdown checkpoint — by rendered
 * transcript bytes, or by the visible protocol transcript when a live
 * responses/thinking binding covers the frontier. A returning client binds to
 * any free slot and restores from disk there; no extra routing metadata is
 * needed because the kvstore's text-prefix index IS the metadata. Live
 * tool-state continuations of an evicted slot get the protocol's honest 409
 * (their sampled frontier is gone), exactly like a server restart.
 *
 * Slot 0 is PINNED — never evicted: (a) client threads read
 * ds4_session_ctx(s->slots[0].sess) lock-free (http_server.c) under the
 * CUDA-state audit's immutable-after-startup exception, so freeing that
 * session would be a data race; (b) slot 0 is the only slot guaranteed to fit
 * any admissible request (job_needed_ctx caps at its ctx), which preserves
 * the scheduler invariant that an all-idle pool always binds the head (the
 * worker sleeps on the condvar when nothing is active) and guarantees the
 * pool never reaches zero slots; (c) as the largest session it is the most
 * expensive one to bring back. When small sessions need room, evicting the
 * idle secondaries (this code) frees the same budget without touching it.
 *
 * Like lazy provisioning, eviction is a deliberate multi-second quantum
 * overshoot on the single GPU worker thread: the snapshot forces a full
 * device sync + a multi-GiB D2H copy + a disk write, and ds4_session_free
 * tears down the graph. It happens only at a scheduling boundary (bind time),
 * and only when the alternative is queueing forever.
 * ========================================================================= */

uint64_t server_ledger_release(uint64_t committed_total, uint64_t slot_cost) {
    if (slot_cost > committed_total) {
        server_log(DS4_LOG_WARNING,
                   "ds4-server: EVICTION LEDGER UNDERFLOW: releasing %.2f GiB from "
                   "%.2f GiB committed — provision/evict pairing is out of sync; "
                   "clamping the ledger to 0 (the MemAvailable floor remains the "
                   "backstop against the resulting over-admission)",
                   (double)slot_cost / (1024.0 * 1024.0 * 1024.0),
                   (double)committed_total / (1024.0 * 1024.0 * 1024.0));
        return 0;
    }
    return committed_total - slot_cost;
}



int server_evict_pick_victim(const session_slot *slots, int n_slots,
                             const bool *protect) {
    int victim = -1;
    for (int i = 1; i < n_slots; i++) { /* slot 0 pinned (see block comment) */
        const session_slot *sl = &slots[i];
        if (!sl->sess || sl->active_job) continue;
        if (protect && protect[i]) continue;
        if (victim < 0 ||
            sl->last_serviced_us < slots[victim].last_serviced_us ||
            (sl->last_serviced_us == slots[victim].last_serviced_us &&
             sl->est_cost_bytes < slots[victim].est_cost_bytes))
        {
            victim = i;
        }
    }
    return victim;
}



/* Mark slots some QUEUED live-tool-state continuation still needs: that KV
 * frontier exists only on its owner slot, so evicting it would turn the
 * queued job into a 409 the moment it binds. The queue is snapshotted under
 * mu; the job pointers stay valid afterwards because only this worker pops
 * jobs and each client thread blocks on its job condvar until then. The
 * owner lookups (tool_mu + session pos) run after mu is released — the two
 * locks are never nested. */
static void worker_protect_queued_owner_slots(server *s,
                                              bool protect[DS4_SESSION_POOL_CAP]) {
    memset(protect, 0, sizeof(protect[0]) * DS4_SESSION_POOL_CAP);
    job *queued[DS4_SERVER_MAX_CLIENTS];
    int n = 0;
    pthread_mutex_lock(&s->mu);
    for (job *q = s->head; q && n < DS4_SERVER_MAX_CLIENTS; q = q->next) {
        queued[n++] = q;
    }
    pthread_mutex_unlock(&s->mu);
    for (int i = 0; i < n; i++) {
        const request *r = &queued[i]->req;
        session_slot *owner = NULL;
        if (r->api == API_RESPONSES && r->responses_live_call_ids.len > 0) {
            owner = responses_live_slot_for_ids(s, &r->responses_live_call_ids);
        } else if (r->api == API_ANTHROPIC &&
                   r->anthropic_live_call_ids.len > 0) {
            owner = anthropic_live_slot_for_ids(s, &r->anthropic_live_call_ids);
        }
        if (owner) protect[owner - s->slots] = true;
    }
}



/* Pointless-eviction guard #2: evicting is only worth its cost if releasing
 * idle sessions can actually admit the provisioning the head job needs.
 * If even reclaiming EVERY unprotected idle slot leaves admission refusing,
 * skip eviction entirely — the head is genuinely waiting for a busy slot to
 * free, and evicting warm idle sessions would only churn snapshots. (Host
 * arithmetic only: ds4_engine_session_cost_bytes is the same sizing code the
 * allocator uses, no CUDA work; runs only on failed bind attempts. Guard #1
 * is the provisioning-refusal reason check in worker_try_bind.) */
static bool worker_eviction_could_help(server *s, const job *j,
                                       const bool *protect) {
    const uint64_t est =
        ds4_engine_session_cost_bytes(s->engine, provision_ctx_for_job(s, j));
    if (est == 0) return false;
    /* Model the MemAvailable floor too (2026-07-15 review): a POOL_FULL
     * refusal never consulted it, so on a physically tight box the gate
     * could open, one warm session be evicted, and the re-provisioning then
     * refuse on MEM_FLOOR anyway — the exact churn the refusal-reason gate
     * exists to prevent. Eviction does not promptly raise MemAvailable (see
     * the block comment above), so if the floor refuses NOW it will refuse
     * after the eviction too; skip. Fail closed on an unreadable gauge,
     * matching provision_slot. One /proc read per failed bind attempt. */
    const uint64_t avail = server_mem_available_bytes();
    if (avail < est + DS4_SERVER_MEM_FLOOR_BYTES) return false;
    uint64_t reclaimable = 0;
    bool any = false;
    for (int i = 1; i < s->n_slots; i++) {
        const session_slot *sl = &s->slots[i];
        if (!sl->sess || sl->active_job || (protect && protect[i])) continue;
        reclaimable += sl->est_cost_bytes;
        any = true;
    }
    if (!any) return false;
    pthread_mutex_lock(&s->mu);
    const uint64_t committed = s->kv_committed_bytes;
    pthread_mutex_unlock(&s->mu);
    const uint64_t after = committed > reclaimable ? committed - reclaimable : 0;
    return server_kv_admits(s->kv_budget_bytes, after, est);
}



/* Evict one idle slot (LRU victim): snapshot to the disk kv cache when
 * possible, free the session, release the ledger, and leave the slot entry
 * (sess == NULL) for provision_slot to reuse. Failure honesty: a failed or
 * unavailable snapshot only costs the returning client a re-prefill — the
 * eviction itself proceeds, and the response always belongs to the right
 * conversation because the freed KV can never be read again. Returns false
 * when nothing is evictable. Worker thread only. */
static bool worker_evict_one(server *s, bool protect[DS4_SESSION_POOL_CAP]) {
    const int vi = server_evict_pick_victim(s->slots, s->n_slots, protect);
    if (vi < 0) return false;
    session_slot *sl = &s->slots[vi];

    const ds4_tokens *tokens = ds4_session_tokens(sl->sess);
    const int live_tokens = tokens ? tokens->len : 0;
    bool stored = false;
    if (s->kv.enabled && live_tokens >= s->kv.opt.min_tokens) {
        stored = kv_cache_store_current(s, sl, "evict");
    }
    if (!stored && live_tokens > 0) {
        server_log(DS4_LOG_WARNING,
                   "ds4-server: slot %d evicting WITHOUT a disk snapshot (%s); "
                   "a returning client re-prefills (correctness unaffected)",
                   vi,
                   !s->kv.enabled ? "kv disk cache disabled"
                   : live_tokens < s->kv.opt.min_tokens
                       ? "conversation below kv-cache min-tokens"
                       : "snapshot write failed");
    }
    /* The live protocol bindings describe a sampled frontier that is about to
     * lose its GPU state; clear them AFTER the store (snapshot keying reads
     * them). A later continuation of those ids gets the protocol's 409. */
    responses_live_clear(s, sl);
    anthropic_live_clear(s, sl);
    thinking_live_clear(s, sl);

    /* Free the session and verify, against the allocator's ground-truth
     * counter, that the bytes the ledger is about to release actually came
     * back (the create-side twin of server_reconciled_session_cost). */
    const uint64_t committed = sl->est_cost_bytes;
    const uint64_t alloc_before = ds4_gpu_tensor_alloc_bytes_current();
    ds4_session_free(sl->sess);
    const uint64_t alloc_after = ds4_gpu_tensor_alloc_bytes_current();
    const uint64_t freed =
        alloc_before > alloc_after ? alloc_before - alloc_after : 0;
    if (committed > 0 &&
        (freed > committed + committed / 10 ||
         committed > freed + freed / 10))
    {
        server_log(DS4_LOG_WARNING,
                   "ds4-server: EVICTION FREED-BYTES DRIFT >10%%: ledger releases "
                   "%.2f GiB but the allocator freed %.2f GiB — session teardown "
                   "is leaking or freeing unaccounted allocations",
                   (double)committed / (1024.0 * 1024.0 * 1024.0),
                   (double)freed / (1024.0 * 1024.0 * 1024.0));
    }
    const int evicted_ctx = sl->ctx_size;
    sl->sess = NULL;
    sl->gen = NULL;
    sl->active_job = NULL;
    sl->state = SLOT_EVICTED;
    sl->ctx_size = 0;
    sl->est_cost_bytes = 0;
    sl->tokens_emitted = 0;
    sl->last_serviced_us = 0;
    sl->continued_last_store_tokens = 0;
    pthread_mutex_lock(&s->mu);
    s->kv_committed_bytes = server_ledger_release(s->kv_committed_bytes, committed);
    const uint64_t committed_now = s->kv_committed_bytes;
    pthread_mutex_unlock(&s->mu);
    protect[vi] = true; /* freed hole; never a candidate again this round */
    server_log(DS4_LOG_DEFAULT,
               "ds4-server: evicted slot %d ctx=%d tokens=%d snapshot=%s "
               "released=%.2f GiB (allocator freed %.2f GiB) "
               "committed now %.2f / %.2f GiB, MemAvailable %.2f GiB",
               vi, evicted_ctx, live_tokens, stored ? "disk" : "none",
               (double)committed / (1024.0 * 1024.0 * 1024.0),
               (double)freed / (1024.0 * 1024.0 * 1024.0),
               (double)committed_now / (1024.0 * 1024.0 * 1024.0),
               (double)s->kv_budget_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)server_mem_available_bytes() / (1024.0 * 1024.0 * 1024.0));
    return true;
}



/* Bind the head job to a slot if routing allows it. Strict FIFO: when the
 * head must wait (its owner slot is busy, or no fitting slot is free), later
 * jobs wait behind it — simple and starvation-free. Returns true when the
 * head was consumed: bound to a slot, or failed explicitly (a continuation
 * that cannot fit its owner slot — see choose_slot_for_job). When the head
 * cannot be placed cleanly (nothing fits, or only a warm slot it would
 * clobber), it is not waiting on a busy owner, and the provisioning refusal
 * is one eviction can relieve (full pool / full ledger — never the
 * MemAvailable floor, see the increment-4 block above), idle slots are
 * evicted LRU-first until the head binds without clobbering or eviction
 * stops helping — then the clobber fallback binds it exactly like the
 * increment-3 scheduler did. */
static bool worker_try_bind(server *s) {
    pthread_mutex_lock(&s->mu);
    job *j = s->head; /* peek: only the worker pops */
    pthread_mutex_unlock(&s->mu);
    if (!j) return false;

    int reject_ctx = 0;
    bool waiting_owner = false;
    bool clobbers = false;
    provision_refusal refusal = PROVISION_OK;
    session_slot *sl = choose_slot_for_job(s, j, &reject_ctx, &waiting_owner,
                                           &clobbers, &refusal);
    if ((!sl || clobbers) && !waiting_owner && reject_ctx == 0 &&
        (refusal == PROVISION_REFUSED_POOL_FULL ||
         refusal == PROVISION_REFUSED_ADMISSION))
    {
        bool protect[DS4_SESSION_POOL_CAP];
        worker_protect_queued_owner_slots(s, protect);
        if (worker_eviction_could_help(s, j, protect)) {
            while ((!sl || clobbers) &&
                   (refusal == PROVISION_REFUSED_POOL_FULL ||
                    refusal == PROVISION_REFUSED_ADMISSION))
            {
                /* Refresh owner protection every iteration (2026-07-15
                 * review): each pass through choose_slot_for_job can stall
                 * for seconds inside ds4_session_create, and a live-tool
                 * continuation enqueued during that stall would be invisible
                 * to a one-shot snapshot — its owner could then be evicted
                 * into an avoidable 409. Holes are re-derived from
                 * sess == NULL (the refresh clears worker_evict_one's
                 * hole marks). */
                worker_protect_queued_owner_slots(s, protect);
                for (int i = 0; i < s->n_slots; i++) {
                    if (!s->slots[i].sess) protect[i] = true;
                }
                if (!worker_evict_one(s, protect)) break;
                sl = choose_slot_for_job(s, j, &reject_ctx, &waiting_owner,
                                         &clobbers, &refusal);
            }
        }
    }
    if (!sl && reject_ctx > 0) {
        /* The job can never run: pop it and send the front door's
         * context_length_exceeded client error (against the owner slot's
         * context), then wake the client thread exactly like
         * worker_finish_slot does for a completed job. */
        pthread_mutex_lock(&s->mu);
        s->head = j->next;
        if (!s->head) s->tail = NULL;
        if (s->n_queued > 0) s->n_queued--;
        pthread_mutex_unlock(&s->mu);
        j->next = NULL;
        http_error_context_length_exceeded(j->fd, s->enable_cors, &j->req,
                                           j->req.prompt.len, reject_ctx);
        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);
        return true;
    }
    if (!sl) return false;

    pthread_mutex_lock(&s->mu);
    s->head = j->next;
    if (!s->head) s->tail = NULL;
    if (s->n_queued > 0) s->n_queued--;
    s->n_generating++;
    pthread_mutex_unlock(&s->mu);
    j->next = NULL;

    generate_job_begin(s, sl, j);
    return true;
}



/* Publish the /metrics snapshots — per-slot KV position/context and the
 * engine spec-decode counters — into plain server fields under mu. Client
 * threads must never call into the engine (CUDA-state audit,
 * ds4_server_internal.h), so the worker exports these at startup (cli_main,
 * before the worker thread runs), after binds, and once per quantum;
 * send_metrics reads only the snapshots. Host-int copies, no GPU work. */
void server_publish_metrics_snapshot(server *s) {
    ds4_spec_metrics m;
    ds4_engine_spec_metrics(s->engine, &m);
    pthread_mutex_lock(&s->mu);
    for (int i = 0; i < s->n_slots; i++) {
        s->m_slot_pos[i] = s->slots[i].sess ? ds4_session_pos(s->slots[i].sess) : 0;
        s->m_slot_ctx[i] = s->slots[i].ctx_size;
    }
    s->m_spec = m;
    pthread_mutex_unlock(&s->mu);
}



/* Detach a finished job from its slot and wake its client thread. */
static void worker_finish_slot(server *s, session_slot *sl) {
    job *j = sl->active_job;
    generate_job_end(s, sl);
    pthread_mutex_lock(&s->mu);
    if (s->n_generating > 0) s->n_generating--;
    pthread_mutex_unlock(&s->mu);
    pthread_mutex_lock(&j->mu);
    j->done = true;
    pthread_cond_signal(&j->cv);
    pthread_mutex_unlock(&j->mu);
}



/* The single GPU worker (increment 3): a round-robin scheduler over the slot
 * pool. Each pass binds queued jobs to free slots (FIFO), then advances ONE
 * runnable slot by one quantum — a prefill chunk, or up to
 * DS4_SERVER_DECODE_QUANTUM_TOKENS decode tokens — and flushes that slot's
 * deferred client bytes. With a single active job this degenerates to the
 * increment-2 loop (quantum after quantum on one slot, with only a queue
 * peek — one mutex op, no GPU work — between quanta), so single-client
 * output is byte-identical. All ds4_session_* and CUDA work stays on this
 * thread. On shutdown the scheduler keeps stepping bound jobs (the decode
 * loop observes g_stop_requested) and drains the queue, exactly like the
 * increment-2 worker.
 *
 * Quantum overshoot: binding may lazily provision a slot, and
 * provision_slot's ds4_session_create is a multi-GiB allocation that can
 * take SECONDS — every bound slot stalls for its duration. Binding may also
 * EVICT idle slots first (increment 4): snapshot-to-disk (full device sync +
 * multi-GiB D2H + disk write) plus session teardown, then the provisioning
 * on top. These are the largest quantum overshoots in the system (larger
 * than the DSpark ≤17-token fused burst) and are deliberate single-thread
 * design: the CUDA-state audit (ds4_server_internal.h) rules out a second
 * GPU thread, and both happen only at a scheduling boundary. */
void *worker_main(void *arg) {
    server *s = arg;
    int rr = 0; /* round-robin cursor: first slot index to consider next */
    for (;;) {
        bool bound = false;
        while (worker_try_bind(s)) bound = true;
        if (bound) server_publish_metrics_snapshot(s);

        int n_active = 0;
        for (int i = 0; i < s->n_slots; i++) {
            if (s->slots[i].active_job) n_active++;
        }
        if (n_active == 0) {
            /* With every slot free, choose_slot_for_job never returns NULL
             * (slot 0 always fits), so an unbound head cannot reach this
             * wait: sleeping on the condvar until new work or shutdown is
             * safe. */
            pthread_mutex_lock(&s->mu);
            while (!s->head && !s->stopping) pthread_cond_wait(&s->cv, &s->mu);
            const bool quit = !s->head && s->stopping;
            pthread_mutex_unlock(&s->mu);
            if (quit) break;
            continue;
        }

        session_slot *sl = NULL;
        for (int k = 0; k < s->n_slots; k++) {
            session_slot *c = &s->slots[(rr + k) % s->n_slots];
            if (c->active_job) {
                sl = c;
                rr = (int)(c - s->slots) + 1;
                break;
            }
        }
        if (!sl) continue; /* unreachable: n_active > 0 */

        if (sl->gen && sl->gen->phase != GEN_DONE) {
            generate_job_step(s, sl);
            if (sl->gen) slot_writer_flush(&sl->gen->writer);
            sl->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
        }
        if (!sl->gen || sl->gen->phase == GEN_DONE) {
            worker_finish_slot(s, sl);
        }
        server_publish_metrics_snapshot(s); /* /metrics: once per quantum */
    }
    return NULL;
}

