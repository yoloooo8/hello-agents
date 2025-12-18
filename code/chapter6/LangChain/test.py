"""
LangChain 综合示例
涵盖关键概念：Chain、Prompt Template、Memory、Agent+Tools、RAG
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. 基础设置 - 创建 LLM 客户端
# ============================================================

from langchain_openai import ChatOpenAI

def create_llm():
    """创建 LLM 客户端（兼容 OpenAI API 规范）"""
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        temperature=0.7,
    )


# ============================================================
# 2. Chain（链）- 把多个步骤串起来
# ============================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def demo_chain():
    """
    Chain 示例：Prompt → LLM → 输出解析

    核心概念：用 | 管道符把组件串联起来
    """
    print("\n" + "="*60)
    print("Demo 1: Chain（链式调用）")
    print("="*60)

    llm = create_llm()

    # 创建 Prompt 模板
    prompt = ChatPromptTemplate.from_template(
        "用一句话解释什么是{topic}，要求通俗易懂"
    )

    # 创建 Chain：prompt → llm → 输出解析
    chain = prompt | llm | StrOutputParser()

    # 执行 Chain
    result = chain.invoke({"topic": "机器学习"})
    print(f"输入：机器学习")
    print(f"输出：{result}")

    return chain


# ============================================================
# 3. Prompt Template（提示模板）- 复杂模板示例
# ============================================================

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def demo_prompt_template():
    """
    Prompt Template 示例：支持多角色、变量、对话历史

    核心概念：
    - from_messages: 多轮对话模板
    - MessagesPlaceholder: 动态插入对话历史
    """
    print("\n" + "="*60)
    print("Demo 2: Prompt Template（提示模板）")
    print("="*60)

    llm = create_llm()

    # 复杂模板：系统提示 + 对话历史 + 用户输入
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}专家，回答要简洁专业。"),
        MessagesPlaceholder(variable_name="history"),  # 对话历史占位符
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    # 模拟对话历史
    history = [
        HumanMessage(content="Python 是什么？"),
        AIMessage(content="Python 是一种高级编程语言，以简洁易读著称。"),
    ]

    result = chain.invoke({
        # prompt中的三个变量占位符
        "role": "Python",
        "history": history,
        "question": "它有什么优点？"
    })

    print(f"角色：Python 专家")
    print(f"历史对话：{len(history)} 轮")
    print(f"当前问题：它有什么优点？")
    print(f"回答：{result}")


# ============================================================
# 4. Memory（记忆）- 自动管理对话历史
# ============================================================

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def demo_memory():
    """
    Memory 示例：自动保存和加载对话历史（LangChain 0.3.x 新方式）

    核心概念：
    - InMemoryChatMessageHistory: 内存中保存对话历史
    - RunnableWithMessageHistory: 给 Chain 添加历史管理能力
    """
    print("\n" + "="*60)
    print("Demo 3: Memory（对话记忆）")
    print("="*60)

    llm = create_llm()

    # 存储所有会话的历史（key 是 session_id）
    store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # 创建带历史的 Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    # 包装成带历史管理的 Chain
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  #获取记忆的函数
        input_messages_key="input",  # 用户输入对应到prompt的哪个变量
        history_messages_key="history", # 历史消息插入到prompt的哪个变量
    )

    # 多轮对话（同一个 session）
    config = {"configurable": {"session_id": "user_001"}}

    print("第 1 轮：")
    response1 = chain_with_history.invoke(
        {"input": "我叫张三，是一名数据分析师"},
        config=config
    )
    print(f"  用户：我叫张三，是一名数据分析师")
    print(f"  AI：{response1[:100]}...")

    print("\n第 2 轮：")
    response2 = chain_with_history.invoke(
        {"input": "我叫什么名字？我是做什么的？"},
        config=config
    )
    print(f"  用户：我叫什么名字？我是做什么的？")
    print(f"  AI：{response2[:100]}...")

    # 查看记忆内容
    history = store["user_001"]
    print(f"\n记忆中保存了 {len(history.messages)} 条消息")


# ============================================================
# 5. Agent + Tools（工具调用）- LLM 自主决定调用什么工具
# ============================================================

from langchain_core.tools import tool

@tool
def search_weather(city: str) -> str:
    """查询城市天气信息"""
    # 模拟天气查询
    weather_data = {
        "北京": "晴天，25°C，空气质量良好",
        "上海": "多云，28°C，有轻微雾霾",
        "广州": "阴天，30°C，可能有雷阵雨",
    }
    return weather_data.get(city, f"{city}：暂无天气数据")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，例如：2+3*4"""
    try:
        # 安全计算（生产环境需要更严格的校验）
        allowed_chars = set("0123456789+-*/(). ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"计算结果：{expression} = {result}"
        return "表达式包含非法字符"
    except Exception as e:
        return f"计算错误：{e}"

@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def demo_agent_tools():
    """
    Agent + Tools 示例：LLM 自主决定调用什么工具（LangChain 0.3.x 方式）

    核心概念：
    - @tool 装饰器定义工具
    - llm.bind_tools() 绑定工具到 LLM
    - 手动处理工具调用循环
    """
    print("\n" + "="*60)
    print("Demo 4: Agent + Tools（工具调用）")
    print("="*60)

    llm = create_llm()
    tools = [search_weather, calculate, get_current_time]

    # 将工具绑定到 LLM（LangChain 0.3.x 方式）
    llm_with_tools = llm.bind_tools(tools)

    # 创建工具名称到函数的映射
    tool_map = {t.name: t for t in tools}

    def run_agent(question: str) -> str:
        """运行一次完整的 Agent 循环"""
        messages = [
            ("system", "你是一个智能助手，可以使用工具来帮助用户。根据用户问题选择合适的工具。"),
            ("human", question)
        ]

        # 调用 LLM（可能返回工具调用）
        response = llm_with_tools.invoke(messages)

        # 检查是否有工具调用
        if response.tool_calls:
            # 执行工具调用
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    tool_results.append(f"{tool_name}: {result}")

            # 将工具结果返回给 LLM 生成最终回答
            messages.append(("assistant", str(response.tool_calls)))
            messages.append(("user", f"工具执行结果：{'; '.join(tool_results)}"))
            final_response = llm.invoke(messages)
            return final_response.content
        else:
            return response.content

    # 测试不同问题
    questions = [
        "北京今天天气怎么样？",
        "帮我算一下 (100 + 200) * 3",
        "现在几点了？",
    ]

    for q in questions:
        print(f"\n问题：{q}")
        result = run_agent(q)
        print(f"回答：{result}")


# ============================================================
# 6. RAG（检索增强生成）- 从知识库检索后回答
# ============================================================

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

def demo_rag():
    """
    RAG 示例：从知识库检索相关文档，增强 LLM 回答

    核心流程：
    1. 文档切分 → 向量化 → 存入向量库
    2. 用户问题 → 向量检索 → 找到相关文档
    3. 相关文档 + 问题 → LLM → 回答
    """
    print("\n" + "="*60)
    print("Demo 5: RAG（检索增强生成）")
    print("="*60)

    llm = create_llm()

    # 1. 准备知识库文档
    documents = [
        Document(
            page_content="我们公司的退款政策：购买后 7 天内可无理由退款，需保持商品完好。超过 7 天但在 30 天内，如有质量问题可申请退款。",
            metadata={"source": "退款政策"}
        ),
        Document(
            page_content="配送时效：一线城市（北上广深）次日达，二线城市 2-3 天，其他地区 3-5 天。偏远地区可能需要 7 天以上。",
            metadata={"source": "配送说明"}
        ),
        Document(
            page_content="会员等级：消费满 1000 元升级银卡会员，享 95 折；满 5000 元升级金卡会员，享 9 折；满 20000 元升级钻石会员，享 85 折。",
            metadata={"source": "会员制度"}
        ),
        Document(
            page_content="客服工作时间：周一至周五 9:00-18:00，周末 10:00-17:00。紧急问题可拨打 24 小时热线 400-123-4567。",
            metadata={"source": "客服信息"}
        ),
    ]

    # 2. 创建向量库（使用 FAISS）
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )

    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # 检索 top 2

    # 3. 创建 RAG Chain
    rag_prompt = ChatPromptTemplate.from_template("""
根据以下知识库内容回答用户问题。如果知识库中没有相关信息，请如实告知。

知识库内容：
{context}

用户问题：{question}

回答：""")

    def format_docs(docs):
        return "\n\n".join(f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 4. 测试 RAG
    questions = [
        "买了东西不想要了能退吗？",
        "我在上海，大概几天能收到货？",
        "怎么才能享受 9 折优惠？",
    ]

    for q in questions:
        print(f"\n问题：{q}")
        answer = rag_chain.invoke(q)
        print(f"回答：{answer}")


# ============================================================
# 7. 综合示例：带记忆的 RAG Agent
# ============================================================

def demo_rag_agent_with_memory():
    """
    综合示例：RAG + Agent + Memory（LangChain 0.3.x 方式）

    实际应用中最常见的组合：
    - RAG：从知识库获取专业知识
    - Agent：自主决定是否需要检索
    - Memory：记住对话上下文（手动管理）
    """
    print("\n" + "="*60)
    print("Demo 6: 综合示例（RAG + Agent + Memory）")
    print("="*60)

    llm = create_llm()

    # 准备知识库
    documents = [
        Document(page_content="Python 3.12 新特性：更快的解释器、改进的错误消息、新的类型提示语法。"),
        Document(page_content="Python 最佳实践：使用虚拟环境、遵循 PEP8 规范、编写单元测试、使用类型提示。"),
        Document(page_content="Python 常用库：数据分析用 pandas，机器学习用 scikit-learn，深度学习用 PyTorch。"),
    ]

    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # 创建检索工具
    @tool
    def search_knowledge(query: str) -> str:
        """从知识库搜索相关信息"""
        docs = retriever.invoke(query)
        if docs:
            return f"知识库结果：{docs[0].page_content}"
        return "知识库中未找到相关信息"

    tools = [search_knowledge, get_current_time]

    # 将工具绑定到 LLM
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    # 手动管理对话历史
    chat_history = []

    def run_agent_with_history(question: str) -> str:
        """运行带历史的 Agent"""
        messages = [
            ("system", "你是 Python 技术顾问。可以使用工具搜索知识库获取信息。"),
        ]
        # 添加历史
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                messages.append(("human", msg.content))
            else:
                messages.append(("assistant", msg.content))
        # 添加当前问题
        messages.append(("human", question))

        # 调用 LLM
        response = llm_with_tools.invoke(messages)

        # 处理工具调用
        if response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    tool_results.append(f"{tool_name}: {result}")

            messages.append(("assistant", str(response.tool_calls)))
            messages.append(("user", f"工具执行结果：{'; '.join(tool_results)}"))
            final_response = llm.invoke(messages)
            return final_response.content
        else:
            return response.content

    # 多轮对话测试
    conversations = [
        "Python 3.12 有什么新特性？",
        "有哪些最佳实践推荐？",
        "你刚才说的第一个新特性是什么？",  # 测试记忆
    ]

    for q in conversations:
        print(f"\n用户：{q}")
        result = run_agent_with_history(q)
        # 更新历史
        chat_history.append(HumanMessage(content=q))
        chat_history.append(AIMessage(content=result))
        print(f"助手：{result}")


# ============================================================
# 主程序
# ============================================================

def main():
    """运行所有示例"""
    print("LangChain 综合示例")
    print("=" * 60)

    try:
        # 1. Chain 示例
        # demo_chain()

        # # 2. Prompt Template 示例
        # demo_prompt_template()

        # # 3. Memory 示例
        # demo_memory()

        # # 4. Agent + Tools 示例
        # demo_agent_tools()

        # # 5. RAG 示例
        demo_rag()

        # # 6. 综合示例
        # demo_rag_agent_with_memory()

        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
