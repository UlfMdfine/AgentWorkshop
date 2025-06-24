# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Task list

# %% [markdown]
# ### To get yourself familiar with the agent system and the general functionality, you can use the following prompt.
# ### It does not create a report but only presents the agents. Also, you can see the general workflow. 
#  - Data Management tab: The dataset is uploaded and a sample is presented to get a feeling about the columns and their values.
#  - Report Generation tab: Here, you can write prompts and run them. If a report is created, it is presented here - you may export it. (Results and graphs are also stored in the output folder of the repository.)
#  - Thinking Process tab: Shows the individual steps of the agent and a summary or specific results in the last message. Each call of a function and the used input parameters are shown. This is particular useful for debugging in case any errors are encountered. The workflow should always be checked to identify issues or unwanted side effects.

# %%
"Please do not create a report, only answer the following requests: Please list all the agents you have access to. What is the purpose of each agent? List all the tools each agent has access to."

# %% [markdown]
# ### Now, lets try the first prompt that creates a report including a distribution analysis and a discriminatory power analysis.
# You can run the following prompt and check the thinking process to see what happens. The resulting report shows not only the calculated values/graphs but also the automatically created comments.

# %%
"""
First, perform a ROE distribution plot for the given CSV file. Plot the distribution of the credit rating, weight by exposure and group by employment status.
Plot the discriminatory power using Somers' D as the method. For the time variable use the reporting date and for the target use the default flag.
"""

# %% [markdown]
# ### Try to write your own prompt next. The target is to get a migration matrix for the year 2022.
# - In case you do not see a report, check the thinking process and the input parameters in particular for potential issue.
# - As a second step, lets try the same prompt but in a different language of your choosing.

# %% [markdown]
# ### Lets assume a colleague approaches you asking for help with an analysis.
# They tell you that they experimented with the tool but their prompt does not return a report.
# You tell them that you will look into the thinking process to identify the problem and to then send a fixed prompt.
# Hint: The demo repository provides functionality to determine the unique values of a column of the dataset.

# %%
"Please calculate the PSI using the first of June of 2024. Additionally, please explain the PSI and for what it is used."

# %% [markdown]
# ### You have identified the problem and your colleague thanks you. 
# You show them how to find the error message in the thinking process so that they can debug themselves the next time.
# You are asking yourself, if you could maybe enhance the error message of the PSI calculation function so that other colleagues would not need
# to run an additional prompt like you did but the "fix" is already clearly returned in the thinking process.

# %% [markdown]
# ### To conclude, you are should compile a comprehensive report. Gather and integrate all previous prompts to create a full and final version.
