import os, sys
from pathlib import Path

from deepagents import create_deep_agent
# 所有的 backends
# https://docs.langchain.com/oss/python/deepagents/backends
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langchain.tools import tool
# 常用的消息类型
from langchain.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, AIMessageChunk
# 利用 checkpointer 来提供短期记忆能力
from langgraph.checkpoint.memory import InMemorySaver
# 中间件, 用于总结历史对话内容
from langchain.agents.middleware import SummarizationMiddleware

# 默读取当前目录下的 .env 文件, 可以通过 dotenv_path 来修改
from dotenv import load_dotenv
# 这种写法可以规避在 vscode 中找不到 .env 的问题 (获取项目根目录路径)
project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

# 记录日志
from loguru import logger
logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL"))


# 定义一个 checkpointer (保存在内存中)
checkpointer = InMemorySaver()
# 如果需要会话隔离, 可以传递一个 config 参数, 里面需要指定 thread_id
config = {"configurable": {"thread_id": "1"}}

# 创建 ChatModel: v1 创建聊天模型的方法
chat_model = init_chat_model(
    model_provider="openai",
    # model 也可以写为 <model_provider>:<model_name> 的形式
    # 这样就可以不用指定 model_provider 这个参数了
    model = os.getenv("LMSTUDIO_LLM_MODEL"),
    base_url = os.getenv("LMSTUDIO_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens = 16384,
)

# 创建 Agent
agent = create_deep_agent(
    model=chat_model,
    # 提示词可以不写, 这样也可以在调用时用 SystemMessage 来指定
    system_prompt="你是一个专业的AI助手, 请用简洁的方式回复用户的问题.",
    # 指定 checkpointer, 用于保存 StateGraph 和 短期记忆
    checkpointer=checkpointer,
    # 利用中间件来设定如何总结历史内容
    middleware=[
        SummarizationMiddleware(
            model=chat_model,
            # 触发条件, 可以有多种写法
            # ("messages", 50) -> 当 messages 数量达到50条时触发摘要生成
            # ("tokens", 3000) -> 当tokens 达到3000个时触发摘要生成
            # [("fraction", 0.8), ("messages", 100)] -> 当模型 max_tokens 达到 80% 或 messages 数量达到100条时触发摘要生成
            trigger=("tokens", 25600),
            # 触发摘要后的上下文保存策略
            # ("messages", 20) -> 保存最近20条 messages
            # ("tokens", 3000) -> 保存最近 3000 tokens
            # ("fraction", 0.3) -> 保存模型 max_tokens 的 30%
            keep=("messages", 8),
        ),
    ],
    # 文件系统
    # https://docs.langchain.com/oss/python/deepagents/backends
    backend=FilesystemBackend(
        root_dir=os.path.join(project_root, "90.Example/DeepAgentsDir"),
        # 是否使用虚拟文件系统, 如果为 True 则模型会认为此目录为根目录
        virtual_mode=True,
    ),
    # Skills
    # https://docs.langchain.com/oss/python/deepagents/skills#filesystembackend
    # 该目录必须在 FilesystemBackend 之内 (virtual_mode=True)
    skills=[os.path.join(project_root, "90.Example/DeepAgentsDir/skills")],
    interrupt_on={
        "write_file": True,  # Default: approve, edit, reject
        "read_file": False,  # No interrupts needed
        "edit_file": True    # Default: approve, edit, reject
    },
)

if __name__ == "__main__":
    # Init
    print("欢迎使用 DeepAgents, 如果想退出请输入 exit / quit 即可.\n")

    # 运行智能体
    while True:
        input_text = input("请输入: ")
        
        if input_text.lower() in ["exit", "quit"]:
            logger.info(f"Exiting...")
            break
        elif input_text in ["", "\n"]:
            logger.info(f"忽略输入为空, 跳过...")
            continue
        
        response = agent.invoke(
            {"messages": [HumanMessage(content=input_text)]},
            config=config,
        )

        for message in response["messages"]:
            message.pretty_print()
        
        logger.debug(f"当前消息条数: {len(response["messages"])}")