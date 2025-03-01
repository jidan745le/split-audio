# import sys
# sys.path.append("/path/from/pip-show-command")  # 替换为步骤3得到的路径

from dotenv import load_dotenv
load_dotenv()  # 在文件开头添加这2行

# 导入所需库
from langchain import hub
from langchain_openai import OpenAI,ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.callbacks import StdOutCallbackHandler

from langchain_community.llms import ChatGLM  # ← 正确导入
chatglm = ChatGLM(
    endpoint_url="http://localhost:8000/v1/chat/completions",
    max_tokens=8096,
    temperature=0
)

print(chatglm)
    
# 1. 设置大模型
llm = OpenAI(
    temperature=0,
)

llmChat = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo"  # 或 "gpt-4"
)

print(f"当前使用模型: {llm.model_name}")  # 输出：text-davinci-003

response = llm.invoke("Answer the following questions as best you can. You have access to the following tools:\n\nSearch(query: str, **kwargs: Any) -> str - 当需要回答需要实时数据或知识库中没有的问题 时使用\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [Search]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: 关于kigurumi的知识\nThought:")

print("response: ",response)

# 2. 初始化搜索工具
search = SerpAPIWrapper()  # 自动读取SERPAPI_API_KEY

# 3. 定义工具集
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="当需要回答需要实时数据或知识库中没有的问题时使用"
    ),
]

# 4. 获取ReAct提示模板
react_prompt = hub.pull("hwchase17/react")

print("111")
print(react_prompt)
print("222")

# 5. 创建ReAct Agent
agent = create_react_agent(llm, tools, react_prompt)
 
# 6. 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    callbacks=[StdOutCallbackHandler()]  # 显示详细提示词生成过程
)

# 7. 运行Agent
# response = agent_executor.invoke(
#     {"input": "当前Agent最新研究进展是什么？请用中文回答"}
# )
# print(response["output"]) 

response = agent_executor.invoke(
    {"input": "关于kigurumi的知识"}
)

print(response["output"]) 
