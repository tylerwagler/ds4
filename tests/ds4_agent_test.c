#define DS4_AGENT_TEST
#define DS4_AGENT_TEST_NO_MAIN
#include "../src/agent/config_types.c"
#include "../src/agent/util_cli.c"
#include "../src/agent/prompt_queue.c"
#include "../src/agent/trace.c"
#include "../src/agent/dsml_parser.c"
#include "../src/agent/markdown.c"
#include "../src/agent/tool_viz.c"
#include "../src/agent/progress.c"
#include "../src/agent/kvstore_session.c"
#include "../src/agent/session_ui.c"
#include "../src/agent/file_tools.c"
#include "../src/agent/edit_tools.c"
#include "../src/agent/web_tools.c"
#include "../src/agent/bash_jobs.c"
#include "../src/agent/tool_dispatch.c"
#include "../src/agent/compaction.c"
#include "../src/agent/worker.c"
#include "../src/agent/worker_sync.c"
#include "../src/agent/terminal_ui.c"
#include "../src/agent/runtime.c"

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
