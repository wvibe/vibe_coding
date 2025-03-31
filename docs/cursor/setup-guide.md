# Cursor Setup Guide for Vibe Coding Playground

This guide helps you configure Cursor IDE for effective development within the `vibe_coding` project. Following these steps ensures the AI assistant adheres to project standards and your workflow preferences.

## 1. Configure Rule Locations

First, tell Cursor where to find the project-specific rules and your global preferences:

1.  Open the Command Palette (Cmd+Shift+P or Ctrl+Shift+P).
2.  Search for and select "Preferences: Open User Settings (JSON)".
3.  Add or update the following settings:
    ```json
    {
        // ... other settings ...
        "cursor.rules.context": "docs/cursor", // Points to project rules
        "cursor.globalRules": [ // Location of your global instructions
             "~/vibe/vibe_coding/docs/cursor/custom-instructions.mdc" // Adjust path if needed
        ]
        // ... other settings ...
    }
    ```
    *   **Note:** Ensure the path in `cursor.globalRules` correctly points to the `custom-instructions.mdc` file within your cloned repository location. You might need to adjust `/home/wmu/` if your home directory differs.

## 2. Review Project Rules

The `docs/cursor` directory contains several important files for configuring Cursor:

*   `coding-pref.mdc`: General coding preferences and style guidelines.
*   `my-stack.mdc`: Defines the technology stack and environment details.
*   `custom-instructions.mdc`: Global instructions for the AI assistant.
*   `mcp.json`: Model Configuration Protocol (MCP) server configurations.

Familiarize yourself with these rules. Cursor will use them automatically when working on files within this project.

## 3. Configure Auto-Run Mode

For a smoother workflow, you can allow Cursor to perform certain actions automatically. Configure these settings carefully based on your comfort level.

1.  Open Cursor Settings (Cmd+, or Ctrl+,).
2.  Search for "Cursor Agent".
3.  Under the "Agent" section, configure the following based on the recommended setup:
    *   **[✓] Enable auto-run mode:** Check the box "Allow Agent to run tools without asking for confirmation...".
    *   **Command allowlist:** Add commands you trust the agent to run automatically. A recommended starting list is:
        *   `python`, `git`, `ls`, `mkdir`, `npx`, `find`, `uv`, `cd`, `ruff`, `pytest`, `grep`
    *   **Command denylist:** Add commands that should *never* be run automatically. A recommended starting list includes potentially destructive or sensitive commands:
        *   `sudo`, `passwd`, `rm -rf /`, `rm -rf ~`, `rm -rf .*`, `diskutil`, `kill`, `shutdown`, `reboot`, `iptables`, `curl`, `wget`, `chmod`, `chown`, `shred`, `:(){ :|: & };:`
    *   **[✓] Delete file protection:** Check this box to prevent accidental automatic file deletion.
    *   **[✓] MCP tools protection:** Check this box to prevent the agent from running MCP tools automatically.

*   **Caution:** Only enable auto-run mode and add commands to the allowlist if you understand the potential risks. Always review the denylist carefully.

## 4. Set Up Model Configuration Profiles (MCPs)

MCPs allow you to create different profiles for the AI, tailoring its behavior and enabling additional capabilities through MCP servers.

1.  Open the Command Palette (Cmd+Shift+P or Ctrl+Shift+P).
2.  Search for and select "Cursor: Configure Model Configuration Profiles".
3.  Create a profile specifically for this project (e.g., named "VibeCoding").
4.  Configure the following MCP servers as defined in `mcp.json`:

    *   **Github Vibe MCP:** Enables GitHub integration.
        - You'll need to replace `<GITHUB_PAT_PLACEHOLDER>` with your actual GitHub Personal Access Token.
    *   **AgentDesk BrowserTools:** Provides web browsing capabilities.
    *   **Weights & Biases:** Integration for ML experiment tracking.
    *   **Brave Search:** Enables web search capabilities.

    ```json
    {
      "mcpServers": {
        "Github Vibe MCP": {
          "command": "npx",
          "args": [
            "-y",
            "@smithery/cli@latest",
            "run",
            "@smithery-ai/github",
            "--config",
            "{\"githubPersonalAccessToken\":\"<GITHUB_PAT_PLACEHOLDER>\"}"
          ]
        },
        "AgentDesk BrowserTools": {
          "command": "npx",
          "args": [
            "@agentdeskai/browser-tools-mcp@1.2.0"
          ]
        },
        "weights_and_biases": {
          "command": "uv",
          "args": [
            "--directory",
            "/Users/wmu/vibe/mcp-server",
            "run",
            "src/mcp_server/server.py"
          ]
        },
        "brave-search": {
          "command": "npx",
          "args": [
            "-y",
            "@modelcontextprotocol/server-brave-search"
          ],
          "env": {
            "BRAVE_API_KEY": "<YOUR_BRAVE_API_KEY>"
          }
        }
      }
    }
    ```

    **Note:** Replace all placeholder values (e.g., `<GITHUB_PAT_PLACEHOLDER>`, `<YOUR_BRAVE_API_KEY>`) with your actual API keys and tokens.

## 5. Development Workflow with Cursor

To work effectively with the AI:

1.  **Plan First:** Before asking the AI to write code or make changes, clearly state the goal and ask for a step-by-step plan.
2.  **Approve Plan:** Review the proposed plan. Ask for clarifications or modifications if needed. Only give explicit approval (e.g., "Yes, proceed with the plan") when satisfied.
3.  **Incremental Implementation:** For larger tasks, ask the AI to implement the plan incrementally, checking in after each significant step.
4.  **Document Progress:** Use the chat history or separate documentation (`docs/`) to keep track of decisions made, approaches taken, and implemented changes.
5.  **Use Todos/Milestones:** Break down larger tasks into smaller milestones or TODO items. You can ask the AI to help generate these based on the plan.

## Additional Resources

*   **Intro to Cursor Features:** [https://www.youtube.com/watch?v=TQsP_PlCY1I](https://www.youtube.com/watch?v=TQsP_PlCY1I&list=PLhBPcrfhLEXuzJRfriP_C6KDf3M0oXmRg&index=4)
*   **Cursor AI Pair Programming:** [https://www.youtube.com/watch?v=v7UcVPO4y3c](https://www.youtube.com/watch?v=v7UcVPO4y3c&list=PLhBPcrfhLEXuzJRfriP_C6KDf3M0oXmRg&index=8)

By following this guide, you'll set up Cursor optimally for contributing to the Vibe Coding Playground.