import datetime

from dotenv import load_dotenv


load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq

from schemas import AnswerQuestion,Reflection,ReviseAnswer

llm = ChatGroq(model='openai/gpt-oss-120b')

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

########################################################################################################
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an expert researcher. Current time :{time}

                1. {first_instruction}
                2. Reflect and critique your answer . Be severe to maximize improvement .
                3. Recommend search queries to research information and improve your answer .
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(
    time=lambda : datetime.datetime.now().isoformat()
)

#################################################################################################

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)


first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice="AnswerQuestion"
)

##################################################################################################

revise_instructions = """
Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer .
    - you MUST include numerical citations in your revised answer to ensure it can be verified .
    - Add a "references" section to the bottom of your answer (which doesnt not count towards the word limit)
        - [1] https://example1.com
        - [2] https://example2.com
    - You should use the previous critique to remove superfluous information from your answer to make SURE it is not more than 250 words.
"""

##########################################################################

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer],tool_choice='ReviseAnswer')
###########################################################################


if __name__=="__main__":
    human_message = HumanMessage(
        content="Write about AI Powered SOC / autonomous soc problem domain ,"
        "List startups that do that and raised capital"
    )

    chain = (
        first_responder_prompt_template
        |
        llm.bind_tools(tools=[AnswerQuestion],tool_choice="AnswerQuestion")
        |
        parser_pydantic
    )

    res = chain.invoke({
        "messages":[human_message],
    })

    print(res)