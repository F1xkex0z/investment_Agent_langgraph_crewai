from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.news_crawler import get_stock_news
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint, log_llm_interaction
import json
from datetime import datetime, timedelta
from src.tools.openrouter_config import get_chat_completion
from langchain_core.messages import HumanMessage  # 添加缺失的导入

# 设置日志记录
logger = setup_logger('macro_analyst_agent')


# 添加time模块导入
import time


@agent_endpoint("macro_analyst", "宏观分析师，分析宏观经济环境对目标股票的影响")
def macro_analyst_agent(state: AgentState):
    """Responsible for macro analysis"""
    show_workflow_status("Macro Analyst")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    logger.info(f"正在进行宏观分析: {symbol}")

    # 获取 end_date 并传递给 get_stock_news
    end_date = data.get("end_date")  # 从 run_hedge_fund 传递来的 end_date

    # 获取大量新闻数据（最多100条），传递正确的日期参数
    news_list = get_stock_news(symbol, max_news=100, date=end_date)

    # 过滤七天前的新闻（只对有publish_time字段的新闻进行过滤）
    cutoff_date = datetime.now() - timedelta(days=7)
    recent_news = []
    for news in news_list:
        if 'publish_time' in news:
            try:
                news_date = datetime.strptime(
                    news['publish_time'], '%Y-%m-%d %H:%M:%S')
                if news_date > cutoff_date:
                    recent_news.append(news)
            except ValueError:
                # 如果时间格式无法解析，默认包含这条新闻
                recent_news.append(news)
        else:
            # 如果没有publish_time字段，默认包含这条新闻
            recent_news.append(news)

    logger.info(f"获取到 {len(recent_news)} 条七天内的新闻")

    # 如果没有获取到新闻，返回默认结果
    if not recent_news:
        logger.warning(f"未获取到 {symbol} 的最近新闻，无法进行宏观分析")
        message_content = {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": "未获取到最近新闻，无法进行宏观分析"
        }
    else:
        # 获取宏观分析结果
        logger.info("开始调用宏观新闻分析函数")
        try:
            macro_analysis = get_macro_news_analysis(recent_news)
            message_content = macro_analysis
            logger.info("宏观新闻分析函数调用完成")
        except Exception as e:
            logger.error(f"宏观新闻分析函数调用失败: {str(e)}")
            message_content = {
                "macro_environment": "neutral",
                "impact_on_stock": "neutral",
                "key_factors": [],
                "reasoning": f"宏观分析函数调用失败: {str(e)}"
            }

    # 如果需要显示推理过程
    if show_reasoning:
        show_agent_reasoning(message_content, "Macro Analysis Agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    # 创建消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="macro_analyst_agent",
    )

    show_workflow_status("Macro Analyst", "completed")
    logger.info(f"--- DEBUG: macro_analyst_agent COMPLETED ---")
    logger.info(
        f"--- DEBUG: macro_analyst_agent RETURN messages: {[msg.name for msg in (state['messages'] + [message])]} ---")
    return {
        "messages": state["messages"] + [message],
        "data": {
            **data,
            "macro_analysis": message_content
        },
        "metadata": state["metadata"],
    }


def get_macro_news_analysis(news_list: list) -> dict:
    """分析宏观经济新闻对股票的影响

    Args:
        news_list (list): 新闻列表

    Returns:
        dict: 宏观分析结果，包含环境评估、对股票的影响、关键因素和详细推理
    """
    logger.debug("--- 开始执行 get_macro_news_analysis 函数 ---" + "-"*30)
    
    if not news_list:
        logger.warning("新闻列表为空，返回默认中性分析结果")
        return {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": "没有足够的新闻数据进行宏观分析"
        }

    # 检查缓存
    import os
    cache_file = "src/data/macro_analysis_cache.json"
    logger.debug(f"缓存文件路径: {cache_file}")
    
    # 确保缓存目录存在
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        logger.debug(f"确保缓存目录存在成功: {os.path.dirname(cache_file)}")
    except Exception as e:
        logger.error(f"创建缓存目录失败: {str(e)}")

    # 生成新闻内容的唯一标识
    try:
        news_key = "|" + "|".join([
            f"{news.get('title', '无标题')}|{news.get('publish_time', '未知时间')}"
            for news in news_list[:20]  # 使用前20条新闻作为标识
        ]) + "|"
        logger.debug(f"生成新闻唯一标识 (长度: {len(news_key)}): {news_key[:100]}...")
    except Exception as e:
        logger.error(f"生成新闻唯一标识失败: {str(e)}")
        news_key = str(hash(str(news_list[:20])))  # 使用哈希值作为备选标识
        logger.debug(f"使用哈希值作为备选标识: {news_key}")

    # 检查缓存
    cache = {}
    if os.path.exists(cache_file):
        logger.debug(f"缓存文件存在，尝试读取")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
                logger.debug(f"缓存文件内容大小: {len(file_content)} 字节")
                cache = json.loads(file_content)
                logger.debug(f"缓存文件加载成功，包含 {len(cache)} 条记录")
                if news_key in cache:
                    logger.info("使用缓存的宏观分析结果")
                    logger.debug(f"缓存结果内容: {json.dumps(cache[news_key])}")
                    logger.debug("--- 结束执行 get_macro_news_analysis 函数 (缓存命中) ---" + "-"*30)
                    return cache[news_key]
                else:
                    logger.debug(f"缓存未命中，当前key: {news_key[:50]}...")
        except json.JSONDecodeError as e:
            logger.error(f"缓存文件JSON解析失败: {str(e)}")
            logger.debug(f"缓存文件内容片段: {file_content[:100] if 'file_content' in locals() else '无'}")
            cache = {}
        except Exception as e:
            logger.error(f"读取宏观分析缓存出错: {str(e)}")
            cache = {}
    else:
        logger.info("未找到宏观分析缓存文件，将创建新文件")
        cache = {}

    # 准备系统消息
    logger.debug("开始准备系统消息...")
    system_message = {
        "role": "system",
        "content": """你是一位专业的宏观经济分析师，专注于分析宏观经济环境对A股个股的影响。
        请分析提供的新闻，从宏观角度评估当前经济环境，并分析这些宏观因素对目标股票的潜在影响。
        
        请关注以下宏观因素：
        1. 货币政策：利率、准备金率、公开市场操作等
        2. 财政政策：政府支出、税收政策、补贴等
        3. 产业政策：行业规划、监管政策、环保要求等
        4. 国际环境：全球经济形势、贸易关系、地缘政治等
        5. 市场情绪：投资者信心、市场流动性、风险偏好等
        
        你的分析应该包括：
        1. 宏观环境评估：积极(positive)、中性(neutral)或消极(negative)
        2. 对目标股票的影响：利好(positive)、中性(neutral)或利空(negative)
        3. 关键影响因素：列出3-5个重要的宏观因素
        4. 详细推理：解释为什么这些因素会影响目标股票
        
        请确保你的分析：
        1. 基于事实和数据，而非猜测
        2. 考虑行业特性和公司特点
        3. 关注中长期影响，而非短期波动
        4. 提供具体、可操作的见解"""
    }
    logger.debug(f"系统消息准备完成，长度: {len(system_message['content'])} 字符")

    # 准备新闻内容
    # 减少新闻数量，避免上下文过长
    news_count = min(5, len(news_list))
    logger.info(f"准备 {news_count} 条新闻进行分析")
    
    try:
        news_content = "\n\n".join([
            f"标题：{news.get('title', '无标题')}\n"
            f"来源：{news.get('source', '未知来源')}\n"
            f"时间：{news.get('publish_time', '未知时间')}\n"
            f"内容：{news.get('content', '')[:200]}..."  # 限制内容长度，避免过长
            for news in news_list[:news_count]  # 使用前N条新闻，减少上下文长度
        ])
        logger.debug(f"新闻内容准备完成，总长度: {len(news_content)} 字符")
        logger.debug(f"新闻内容片段: {news_content[:200]}...")
    except Exception as e:
        logger.error(f"准备新闻内容失败: {str(e)}")
        # 返回错误结果
        logger.debug("--- 结束执行 get_macro_news_analysis 函数 (准备新闻内容失败) ---" + "-"*30)
        return {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": f"准备新闻内容失败: {str(e)}"
        }

    user_message = {
        "role": "user",
        "content": f"请分析以下新闻，评估当前宏观经济环境及其对相关A股上市公司的影响：\n\n{news_content}\n\n请以JSON格式返回结果，包含以下字段：macro_environment（宏观环境：positive/neutral/negative）、impact_on_stock（对股票影响：positive/neutral/negative）、key_factors（关键因素数组）、reasoning（详细推理）。"
    }
    logger.debug(f"用户消息准备完成，总长度: {len(user_message['content'])} 字符")

    try:
        # 获取LLM分析结果
        logger.info("正在调用LLM进行宏观分析...")
        messages_for_llm = [system_message, user_message]
        logger.debug(f"发送给LLM的消息数量: {len(messages_for_llm)}")
        logger.debug(f"第一条消息角色: {messages_for_llm[0]['role']}, 长度: {len(messages_for_llm[0]['content'])} 字符")
        logger.debug(f"第二条消息角色: {messages_for_llm[1]['role']}, 长度: {len(messages_for_llm[1]['content'])} 字符")
        
        # 记录调用开始时间
        start_time = time.time()
        result = get_chat_completion(messages_for_llm)
        # 记录调用结束时间和耗时
        end_time = time.time()
        logger.info(f"LLM调用完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 记录返回结果信息
        if result is None:
            logger.error("LLM调用返回None结果")
        else:
            logger.info(f"LLM调用返回结果，长度: {len(result)} 字符")
            logger.debug(f"LLM返回结果前200字符: {result[:200]}...")
            # 记录结果末尾字符，检查是否有截断
            if len(result) > 200:
                logger.debug(f"LLM返回结果后50字符: {result[-50:]}")
            
            # 检查返回结果是否包含JSON格式
            if '{' in result and '}' in result:
                logger.debug("LLM返回结果中检测到可能的JSON格式")
            else:
                logger.warning("LLM返回结果中未检测到JSON格式")

        if result is None:
            logger.error("LLM分析失败，无法获取宏观分析结果")
            logger.debug("--- 结束执行 get_macro_news_analysis 函数 (LLM返回None) ---" + "-"*30)
            return {
                "macro_environment": "neutral",
                "impact_on_stock": "neutral",
                "key_factors": [],
                "reasoning": "LLM分析失败，无法获取宏观分析结果"
            }

        # 解析JSON结果
        logger.debug("开始解析LLM返回的JSON结果...")
        analysis_result = None
        try:
            # 尝试直接解析
            analysis_result = json.loads(result.strip())
            logger.info("成功解析LLM返回的JSON结果")
            logger.debug(f"解析后的JSON结构: {json.dumps(analysis_result, ensure_ascii=False)[:200]}...")
        except json.JSONDecodeError as e:
            logger.warning(f"直接解析JSON失败: {str(e)}")
            # 如果直接解析失败，尝试提取JSON部分
            import re
            logger.debug("尝试从代码块中提取JSON...")
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                logger.debug("找到JSON代码块")
                try:
                    json_content = json_match.group(1).strip()
                    logger.debug(f"提取的JSON内容长度: {len(json_content)} 字符")
                    analysis_result = json.loads(json_content)
                    logger.info("成功从代码块中提取并解析JSON结果")
                    logger.debug(f"从代码块解析的JSON结构: {json.dumps(analysis_result, ensure_ascii=False)[:200]}...")
                except Exception as inner_e:
                    # 如果仍然失败，返回默认结果
                    logger.error(f"无法解析代码块中的JSON结果: {str(inner_e)}")
                    logger.debug(f"尝试解析的代码块内容: {json_content[:200]}..." if 'json_content' in locals() else '无')
                    logger.debug("--- 结束执行 get_macro_news_analysis 函数 (解析代码块JSON失败) ---" + "-"*30)
                    return {
                        "macro_environment": "neutral",
                        "impact_on_stock": "neutral",
                        "key_factors": [],
                        "reasoning": f"无法解析LLM返回的JSON结果: {str(inner_e)}"
                    }
            else:
                # 如果没有找到JSON，返回默认结果
                logger.error("LLM未返回有效的JSON格式结果")
                logger.debug(f"LLM返回的完整结果: {result}")
                logger.debug("--- 结束执行 get_macro_news_analysis 函数 (未找到JSON格式) ---" + "-"*30)
                return {
                    "macro_environment": "neutral",
                    "impact_on_stock": "neutral",
                    "key_factors": [],
                    "reasoning": "LLM未返回有效的JSON格式结果"
                }

        # 检查解析结果的完整性
        required_fields = ["macro_environment", "impact_on_stock", "key_factors", "reasoning"]
        missing_fields = [field for field in required_fields if field not in analysis_result]
        if missing_fields:
            logger.warning(f"解析结果缺少必要字段: {missing_fields}")
            # 补全缺失的字段
            for field in missing_fields:
                if field == "key_factors":
                    analysis_result[field] = []
                else:
                    analysis_result[field] = "neutral"
            logger.debug(f"补全后的分析结果: {json.dumps(analysis_result, ensure_ascii=False)[:200]}...")

        # 缓存结果
        logger.debug("准备缓存分析结果...")
        cache[news_key] = analysis_result
        try:
            logger.debug(f"将结果添加到缓存，当前缓存记录数: {len(cache)}")
            # 限制缓存大小，最多保存100条记录
            if len(cache) > 100:
                logger.debug("缓存记录数超过100，清理最旧的记录")
                # 获取所有键并排序（基于键名，简单实现）
                sorted_keys = sorted(cache.keys())
                # 删除最旧的记录
                for key in sorted_keys[:-100]:
                    del cache[key]
                logger.debug(f"清理后缓存记录数: {len(cache)}")
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.info(f"宏观分析结果已缓存到文件: {cache_file}")
        except Exception as e:
            logger.error(f"写入宏观分析缓存出错: {str(e)}")
            # 即使缓存失败，也继续返回分析结果

        logger.debug("--- 结束执行 get_macro_news_analysis 函数 (成功) ---" + "-"*30)
        return analysis_result

    except Exception as e:
        logger.error(f"宏观分析出错: {str(e)}", exc_info=True)  # 使用exc_info=True记录完整的异常堆栈
        # 尝试记录更多上下文信息以帮助调试
        try:
            import traceback
            logger.debug(f"异常堆栈: {traceback.format_exc()}")
        except:
            pass
        logger.debug("--- 结束执行 get_macro_news_analysis 函数 (异常) ---" + "-"*30)
        return {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": f"分析过程中出错: {str(e)}"
        }
