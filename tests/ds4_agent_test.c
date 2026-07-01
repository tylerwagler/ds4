#define DS4_AGENT_TEST
#define DS4_AGENT_TEST_NO_MAIN
#include "../agent/config_types.c"
#include "../agent/util_cli.c"
#include "../agent/prompt_queue.c"
#include "../agent/trace.c"
#include "../agent/dsml_parser.c"
#include "../agent/markdown.c"
#include "../agent/tool_viz.c"
#include "../agent/progress.c"
#include "../agent/kvstore_session.c"
#include "../agent/session_ui.c"
#include "../agent/file_tools.c"
#include "../agent/edit_tools.c"
#include "../agent/web_tools.c"
#include "../agent/bash_jobs.c"
#include "../agent/tool_dispatch.c"
#include "../agent/compaction.c"
#include "../agent/worker.c"
#include "../agent/worker_sync.c"
#include "../agent/terminal_ui.c"
#include "../agent/runtime.c"

int main(void) {
    ds4_agent_unit_tests_run();
    if (agent_test_failures) {
        fprintf(stderr, "ds4-agent tests: %d failure(s)\n",
                agent_test_failures);
        return 1;
    }
    puts("ds4-agent tests: ok");
    return 0;
}
