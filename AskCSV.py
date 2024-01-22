from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.chat_models import AzureChatOpenAI
import pandas as pd
import openai
# Configure the baseline configuration of the OpenAI library for Azure OpenAI Service.
openai.api_type = "azure"
openai.api_base = "https://bank-hapoalim.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = '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'


def json_to_csv(json_file, csv_file):
    # Load JSON data into a DataFrame
    df = pd.read_json(json_file)
    print(df['cra5e_referencesource@OData.Community.Display.V1.FormattedValue'])

    # Export DataFrame to CSV
    df.to_csv(csv_file, index=False)

    print(f"CSV file '{csv_file}' created successfully.")


# Example usage
json_to_csv('mashavJson.json', 'output.csv')




agent = create_csv_agent(
    AzureChatOpenAI(temperature=0, max_tokens=800, openai_api_base=openai.api_base,
                    openai_api_key='5421003d73ec4e858045932440a8bfa7',
                    openai_api_version=openai.api_version, deployment_name="gpt-4-128k"),
    "C:\\Users\\ltobaly\\PycharmProjects\\pythonProject3\\output.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

print(agent.run("כמה רשומות יש בטבלה"))
