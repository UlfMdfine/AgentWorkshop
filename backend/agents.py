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

commentary_agent = create_react_agent(
    model=llm,
    tools=[summarize_chart],
    prompt=(
        "You are a commentary agent. Your task is to generate descriptions of charts."
    ),
    name="Commentary_Agent",
)

word_agent = create_react_agent(
    model=llm,
    tools=[create_word_document],
    prompt=(
        "You are a word agent. Your task is to create a DOCX report using chart images from a statistics agent and descriptions from a commentary agent."
        "ROE distribution plot files follow the naming scheme '<var_name>'_weighted_by_<weights>.png', e.g. 'Credit_Rating_distribution_weighted_by_Lgd', etc."
        "If you cannot find an image with the name you expect, try to process the image with the logically/contextually closest matching file name."
    ),
    name="Word_Agent",
)


supervisor = create_supervisor(
    model=llm,
    agents=[statistics_agent, commentary_agent, word_agent],
    prompt=(
        "You are a supervisor coordinating plotting, commentary, and report creation.\n"
        "Use the Statistics Agent to generate figures from the CSV as requested by the user.\n"
        "Use the Commentary Agent to describe the generated figures.\n"
        "Required image files are saved in the folder 'outputs', as .png, if you need to access them for the report, and match the summaries automatically."
        "Migration Matrix Bandwidth and Herfindahl-Hirschman Index (HHI) summaries are text-only and have no matching image files."
        "Apply the Migration Matrix Bandwidth summary below the Migration Matrix summary and the HHI summary below the PSI summary."
        "Use the Word Agent to assemble a final report with all png figures in the 'outputs' folder and the matching commentaries. Only use the summaries provided by the Commentary Agent.\n"
        "Make sure the report contains the generated .png figures and save the report as 'AI_Agent_Summary_Final.docx'."
        "If there is an error because of a mismatch of summaries and images, return which images"
        "and summaries you processed and what the mismatch is. Try to run the word agent again after an error in generating the report."
    ),
    supervisor_name="Supervisor",
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()
