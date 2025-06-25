from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from backend.tools import (
    calc_hhi_tool,
    calc_matrix_weighted_bandwidth_tool,
    calc_migration_matrix_tool,
    calc_psi_tool,
    calc_unique_values_tool,
    create_word_document,
    llm,
    plot_migration_matrix_tool,
    plot_psi_tool,
    plot_roe_discriminatory_power_tool,
    plot_roe_distribution_tool,
    summarize_chart,
)

statistics_agent = create_react_agent(
    model=llm,
    tools=[
        plot_roe_distribution_tool,
        plot_roe_discriminatory_power_tool,
        calc_migration_matrix_tool,
        plot_migration_matrix_tool,
        calc_matrix_weighted_bandwidth_tool,
        calc_psi_tool,
        plot_psi_tool,
        calc_hhi_tool,
        calc_unique_values_tool,
    ],
    prompt=(
        "You are a statistics agent. Your task is to calculate statistics from a CSV file and generate charts. Ensure that the charts requested by the user are saved with suitable and relatable file names as .png files in the 'outputs' folder."
    ),
    name="Statistics_Agent",
)

word_agent = create_react_agent(
    model=llm,
    tools=[create_word_document],
    prompt=(
        "You are a word agent. Your task is to create a DOCX report using chart images from a statistics agent and summaries from a commentary agent, if available."
        "ROE distribution plot files follow the naming scheme '<var_name>'_weighted_by_<weights>.png', e.g. 'Credit_Rating_distribution_weighted_by_Lgd', etc."
        "If you cannot find an image with the name you expect, try to process the image with the logically/contextually closest matching file name."
        "If you cannot find a summary, only show the images."
    ),
    name="Word_Agent",
)

# TODO: Uncomment and complete the commentary agent creation. Add the summarize_chart tool to the tools list. Add a prompt that instructs the agent to summarize charts - keep it simple.
# commentary_agent = create_react_agent(
#     model=...,
#     tools=[...],
#     prompt=(
#         "..."
#     ),
#     name="Commentary_Agent",
# )


supervisor = create_supervisor(
    model=llm,
    agents=[
        statistics_agent,
        word_agent,
    ],  # TODO: Add commentary_agent to the list of agents when it is created.
    prompt=(
        "You are a supervisor coordinating plotting, commentary, and report creation.\n"
        "Use the Statistics Agent to generate figures from the CSV as requested. These figures are saved as .png files in the outputs folder.\n"
        # "Use the Commentary Agent to ... .\n" # TODO: Uncomment and complete this line when the commentary agent is created.
        "Use the Word Agent to assemble a final report including:"
        "- All .png figures from the outputs folder."
        "- Only the summaries produced by a Commentary Agent. Never create summaries yourself! If no Commentary Agent is available, send None in place of a summary list.\n"
        "The Migration Matrix Bandwidth and Herfindahl-Hirschman Index (HHI) summaries are text-only and do not have corresponding image files."
        "Position the Migration Matrix Bandwidth summary directly below the Migration Matrix, and the HHI summary below the PSI.\n"
        "Save the report as AI_Agent_Summary_Final.docx."
        "If report generation fails, rerun the Word Agent."
    ),
    supervisor_name="Supervisor",
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()
