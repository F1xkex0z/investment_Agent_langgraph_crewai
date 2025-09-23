"""
数据源接口适配器
集成原系统的真实数据获取功能
"""

import sys
import os
import time
import importlib
import akshare as ak
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import numpy as np
import json

# 添加原系统路径以便导入现有工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from crewai_system.src.utils.logging_config import get_logger, log_info, log_error, log_success, log_failure, log_performance, log_data_collection, log_api_call, log_market_data
from crewai.tools import BaseTool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crewai_system.src.config import config


class DataSourceAdapter:
    """数据源适配器，封装原系统的数据获取功能"""

    def __init__(self):
        # 使用统一日志系统
        self.logger = get_logger("data")
        self.debug_logger = get_logger("debug")
        self._cache = {}
        self._cache_ttl = 300  # 5分钟缓存

        # 尝试导入真实数据源
        self._import_real_data_sources()

    def _log_data_collection(self, data_type: str, identifier: str, data_size: int,
                           execution_time: float, success: bool, error: str = None):
        """记录数据收集性能信息"""
        status = "成功" if success else "失败"
        error_info = f", error={error}" if error else ""
        self.logger.info(f"{data_type}收集{status}: {identifier}, data_size={data_size}, execution_time={execution_time:.2f}s{error_info}")

    def _import_real_data_sources(self):
        """导入真实数据源"""
        start_time = datetime.now()
        self.logger.debug(f"开始导入数据源，时间: {start_time.isoformat()}")

        try:
            # 导入akshare
            import akshare as ak
            self.ak = ak
            self.logger.info("成功导入akshare")
        except ImportError:
            self.logger.error("akshare不可用，系统无法正常运行")
            raise ImportError("akshare不可用，请安装akshare包")

        # 设置请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # 尝试导入原系统的API模块
        try:
            # 添加项目根目录到Python路径
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # 尝试直接导入，绕过Python模块系统
            import importlib.util

            # 构建API模块的完整路径
            api_path = os.path.join(project_root, 'src', 'tools', 'api.py')
            if os.path.exists(api_path):
                self.logger.info(f"找到API文件: {api_path}")
                # 动态导入API模块
                spec = importlib.util.spec_from_file_location("api_module", api_path)
                api_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(api_module)

                get_financial_metrics = api_module.get_financial_metrics
                get_financial_statements = api_module.get_financial_statements
                get_market_data = api_module.get_market_data
                get_price_history = api_module.get_price_history

                self.logger.info("成功导入原系统API模块")
            else:
                self.logger.warning(f"未找到API文件: {api_path}")
                raise ImportError("无法找到API文件")
            self.original_api = {
                'get_financial_metrics': get_financial_metrics,
                'get_financial_statements': get_financial_statements,
                'get_market_data': get_market_data,
                'get_price_history': get_price_history
            }
            self.logger.info("成功导入原系统API模块")
        except ImportError as e:
            self.logger.warning(f"无法导入原系统API模块: {e}")
            self.original_api = None

        # 直接使用akshare获取新闻数据，不再尝试导入新闻爬虫模块
        self.news_crawler = None

        # 设置真实数据获取方法
        self._setup_real_data_methods()

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        self.logger.info("数据源导入完成")

    def _setup_real_data_methods(self):
        """设置真实数据获取方法"""

        # 如果有真实的API函数，优先使用
        if hasattr(self, 'original_api') and self.original_api is not None:
            self.logger.info("使用真实的API函数获取数据")
            self.get_price_history = self.original_api['get_price_history']
            self.get_financial_metrics = self.original_api['get_financial_metrics']
            self.get_market_data = self.original_api['get_market_data']
            self.get_financial_statements = self.original_api['get_financial_statements']

            # 直接使用akshare的新闻搜索函数
            def _direct_search_financial_news(keywords: str, num_articles: int = 10) -> List[Dict[str, Any]]:
                """直接使用akshare获取新闻数据"""
                self.logger.info(f"🔍 [DEBUG] 开始获取新闻数据: keywords={keywords}, num_articles={num_articles}")
                try:
                    # 首先尝试使用东方财富新闻接口
                    if keywords.isdigit() and len(keywords) in [6]:
                        # 直接使用股票代码获取新闻
                        self.logger.info(f"🔍 [DEBUG] 使用东方财富新闻接口获取股票 {keywords} 的新闻")
                        company_news_df = ak.stock_news_em(symbol=keywords)
                        source = "东方财富"
                        self.logger.info(f"🔍 [DEBUG] 东方财富接口返回数据形状: {company_news_df.shape if not company_news_df.empty else '空数据'}")
                        if not company_news_df.empty:
                            self.logger.info(f"🔍 [DEBUG] 东方财富数据列名: {list(company_news_df.columns)}")
                    else:
                        # 对于非股票代码的关键词，使用财联社全球财经快讯
                        self.logger.info(f"🔍 [DEBUG] 使用财联社全球财经快讯接口，关键词: {keywords}")
                        company_news_df = ak.stock_info_global_cls()
                        source = "财联社"
                        self.logger.info(f"🔍 [DEBUG] 财联社接口返回数据形状: {company_news_df.shape if not company_news_df.empty else '空数据'}")
                        if not company_news_df.empty:
                            self.logger.info(f"🔍 [DEBUG] 财联社数据列名: {list(company_news_df.columns)}")

                        # 过滤包含关键词的新闻
                        if not company_news_df.empty:
                            text_columns = ['标题', '内容'] if '内容' in company_news_df.columns else ['标题']
                            self.logger.info(f"🔍 [DEBUG] 可用文本列: {text_columns}")
                            mask = False
                            for col in text_columns:
                                if col in company_news_df.columns:
                                    mask |= company_news_df[col].astype(str).str.contains(keywords, case=False, na=False)
                            company_news_df = company_news_df[mask]
                            self.logger.info(f"🔍 [DEBUG] 关键词过滤后数据形状: {company_news_df.shape if not company_news_df.empty else '空数据'}")

                    # 处理结果
                    news_items = []
                    if not company_news_df.empty:
                        company_news_df = company_news_df.head(num_articles)
                        self.logger.info(f"🔍 [DEBUG] 限制新闻数量为 {num_articles} 条后数据形状: {company_news_df.shape}")

                        for index, row in company_news_df.iterrows():
                            # 根据数据源确定字段名
                            if source == "东方财富":
                                title = row.get('新闻标题', f"关于{keywords}的财经新闻")
                                content = row.get('新闻内容', title)
                                publish_time = row.get('发布时间', datetime.now().isoformat())
                                article_source = row.get('文章来源', source)
                            else:
                                title = row.get('标题', f"关于{keywords}的财经新闻")
                                content = row.get('内容', title)
                                publish_time = row.get('发布时间', datetime.now().isoformat())
                                article_source = source

                            news_item = {
                                "title": title,
                                "content": content,
                                "publish_time": publish_time,
                                "source": article_source,
                                "importance": "high" if index < 3 else "medium"
                            }
                            news_items.append(news_item)

                        self.logger.info(f"🔍 [DEBUG] 成功处理 {len(news_items)} 条新闻")
                        # 记录前几条新闻标题用于调试
                        for i, item in enumerate(news_items[:3]):
                            self.logger.info(f"🔍 [DEBUG] 新闻{i+1}: {item['title'][:50]}... (来源: {item['source']})")
                    else:
                        self.logger.warning(f"🔍 [DEBUG] 未找到任何新闻数据")

                    return news_items
                except Exception as e:
                    self.logger.error(f"🔍 [DEBUG] 获取新闻数据失败: {e}")
                    import traceback
                    self.logger.error(f"🔍 [DEBUG] 详细错误信息: {traceback.format_exc()}")
                    return []
            self.search_financial_news = _direct_search_financial_news
        else:
            self.logger.error("无法导入真实的API模块，系统无法正常运行")
            raise ImportError("无法导入真实的API模块，请检查系统配置")


    def get_cache_key(self, func_name: str, **kwargs) -> str:
        """生成缓存键"""
        cache_data = {
            "func": func_name,
            "kwargs": {k: v for k, v in kwargs.items() if k not in ['password', 'token']}
        }
        return hash(str(sorted(cache_data.items())))

    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        if cache_key in self._cache:
            cached_time, result = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                self.logger.debug(f"使用缓存结果: {cache_key}")
                return result
            else:
                del self._cache[cache_key]
        return None

    def cache_result(self, cache_key: str, result: Dict[str, Any]):
        """缓存结果"""
        self._cache[cache_key] = (time.time(), result)

    def get_market_data_safe(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """安全获取市场数据"""
        start_time = datetime.now()
        self.logger.debug(f"开始获取市场数据，股票代码: {ticker}")
        log_api_call("market_data", "GET", {"ticker": ticker, **kwargs})

        cache_key = self.get_cache_key("get_market_data", ticker=ticker, **kwargs)
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            log_data_collection("market_data_cache", ticker, len(str(cached_result)), execution_time, "从缓存获取")
            self.logger.info(f"从缓存获取市场数据: {ticker}, 耗时: {execution_time:.2f}秒")
            return cached_result

        try:
            result = self.get_market_data(ticker, **kwargs)
            self.cache_result(cache_key, result)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            log_data_collection("market_data_api", ticker, len(str(result)), execution_time, "API调用成功")
            log_market_data(ticker, "akshare_api", len(str(result)))
            self._log_data_collection("市场数据", ticker, len(str(result)), execution_time, True)
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            log_data_collection("market_data_api", ticker, 0, execution_time, f"API调用失败: {str(e)}")
            log_api_call("market_data", "GET", {"ticker": ticker, **kwargs}, execution_time, "FAILED")
            self._log_data_collection("市场数据", ticker, 0, execution_time, False, str(e))
            self.logger.error(f"获取市场数据失败 {ticker}: {e}")
            return {"market_cap": 0, "error": str(e)}

    def get_price_history_safe(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """安全获取价格历史"""
        start_time = datetime.now()
        self.debug_logger.debug(f"获取价格历史开始: ticker={ticker}, start_date={start_date}, end_date={end_date}")
        log_api_call("price_history", "GET", {"ticker": ticker, "start_date": start_date, "end_date": end_date})

        cache_key = self.get_cache_key("get_price_history", ticker=ticker, start_date=start_date, end_date=end_date)
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            result_df = pd.DataFrame(cached_result)
            log_data_collection("price_history_cache", ticker, len(result_df), execution_time, "从缓存获取")
            self.logger.info(f"价格历史获取成功(缓存): ticker={ticker}, data_size={len(result_df)}, execution_time={execution_time:.2f}s, 使用缓存数据")
            return result_df

        try:
            result = self.get_price_history(ticker, start_date, end_date)
            if result is not None and not result.empty:
                # 转换为字典格式缓存
                cache_data = result.to_dict('records')
                self.cache_result(cache_key, cache_data)
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                # 计算价格范围用于日志
                if 'close' in result.columns:
                    price_range = (result['close'].min(), result['close'].max())
                else:
                    price_range = None
                log_data_collection("price_history_api", ticker, len(result), execution_time, "API调用成功")
                log_market_data(ticker, "akshare_api", len(result), price_range)
                self.logger.info(f"价格历史获取成功: ticker={ticker}, data_size={len(result)}, execution_time={execution_time:.2f}s")
            else:
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                log_data_collection("price_history_api", ticker, 0, execution_time, "API返回空数据")
                self.logger.info(f"价格历史获取成功(空数据): ticker={ticker}, data_size=0, execution_time={execution_time:.2f}s, 返回空数据")
            return result if result is not None else pd.DataFrame()
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            log_data_collection("price_history_api", ticker, 0, execution_time, f"API调用失败: {str(e)}")
            log_api_call("price_history", "GET", {"ticker": ticker, "start_date": start_date, "end_date": end_date}, execution_time, "FAILED")
            self.logger.info(f"价格历史获取失败: ticker={ticker}, execution_time={execution_time:.2f}s, error={str(e)}")
            self.logger.error(f"获取价格历史失败 {ticker}: {e}")
            return pd.DataFrame()

    def get_financial_metrics_safe(self, ticker: str) -> Dict[str, Any]:
        """安全获取财务指标"""
        start_time = datetime.now()
        self.debug_logger.debug(f"获取财务指标开始: ticker={ticker}")

        cache_key = self.get_cache_key("get_financial_metrics", ticker=ticker)
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.logger.info(f"财务指标获取成功(缓存): ticker={ticker}, data_size={len(str(cached_result))}, execution_time={execution_time:.2f}s, 使用缓存数据")
            return cached_result

        try:
            result = self.get_financial_metrics(ticker)
            self.cache_result(cache_key, result)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.logger.info(f"财务指标获取成功: ticker={ticker}, data_size={len(str(result))}, execution_time={execution_time:.2f}s")
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.logger.info(f"财务指标获取失败: ticker={ticker}, execution_time={execution_time:.2f}s, error={str(e)}")
            self.logger.error(f"获取财务指标失败 {ticker}: {e}")
            return {}

    def get_financial_statements_safe(self, ticker: str) -> Dict[str, Any]:
        """安全获取财务报表"""
        start_time = datetime.now()
        self.debug_logger.debug(f"获取财务报表开始: ticker={ticker}")

        cache_key = self.get_cache_key("get_financial_statements", ticker=ticker)
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.logger.info(f"财务报表获取成功(缓存): ticker={ticker}, data_size={len(str(cached_result))}, execution_time={execution_time:.2f}s, 使用缓存数据")
            return cached_result

        try:
            result = self.get_financial_statements(ticker)
            self.cache_result(cache_key, result)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.logger.info(f"财务报表获取成功: ticker={ticker}, data_size={len(str(result))}, execution_time={execution_time:.2f}s")
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.logger.info(f"财务报表获取失败: ticker={ticker}, execution_time={execution_time:.2f}s, error={str(e)}")
            self.logger.error(f"获取财务报表失败 {ticker}: {e}")
            return {}

    def search_financial_news_safe(self, keywords: str, num_articles: int = 10) -> List[Dict[str, Any]]:
        """安全搜索财经新闻"""
        start_time = datetime.now()
        self.logger.info(f"🔍 [DEBUG] ===== 开始搜索财经新闻 =====")
        self.logger.info(f"🔍 [DEBUG] keywords={keywords}, num_articles={num_articles}")

        cache_key = self.get_cache_key("search_financial_news", keywords=keywords, num_articles=num_articles)
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.logger.info(f"🔍 [DEBUG] ===== 使用缓存数据 =====")
            self.logger.info(f"🔍 [DEBUG] 缓存数据大小: {len(cached_result)}")
            # 记录缓存中的新闻标题到日志
            if cached_result:
                for i, news_item in enumerate(cached_result[:3]):  # 只记录前3条，避免日志过长
                    title = news_item.get('title', '无标题')
                    source = news_item.get('source', '未知来源')
                    self.logger.info(f"🔍 [DEBUG] 缓存新闻{i+1}: [{source}] {title}")
            self.logger.info(f"🔍 [DEBUG] 财经新闻获取成功(缓存): keywords={keywords}, data_size={len(cached_result)}, execution_time={execution_time:.2f}s")
            return cached_result

        try:
            self.logger.info(f"🔍 [DEBUG] ===== 调用实际新闻搜索方法 ======")
            result = self.search_financial_news(keywords, num_articles)
            self.logger.info(f"🔍 [DEBUG] 新闻搜索方法返回结果大小: {len(result) if result else 0}")

            self.cache_result(cache_key, result)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            self.logger.info(f"🔍 [DEBUG] ===== 新闻搜索完成 =====")
            # 记录获取的新闻标题到日志
            if result:
                self.logger.info(f"🔍 [DEBUG] 开始记录新闻详情:")
                for i, news_item in enumerate(result[:5]):  # 记录前5条
                    title = news_item.get('title', '无标题')
                    source = news_item.get('source', '未知来源')
                    publish_time = news_item.get('publish_time', '未知时间')
                    content_length = len(news_item.get('content', ''))
                    self.logger.info(f"🔍 [DEBUG] 新闻{i+1}: [{source}] {title[:80]}... (发布时间: {publish_time}, 内容长度: {content_length})")
            else:
                self.logger.warning(f"🔍 [DEBUG] 新闻搜索结果为空!")

            self.logger.info(f"🔍 [DEBUG] 财经新闻获取成功: keywords={keywords}, data_size={len(result)}, execution_time={execution_time:.2f}s")
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.logger.error(f"🔍 [DEBUG] ===== 新闻搜索异常 =====")
            self.logger.error(f"🔍 [DEBUG] 财经新闻获取失败: keywords={keywords}, execution_time={execution_time:.2f}s, error={str(e)}")
            self.logger.error(f"🔍 [DEBUG] 搜索财经新闻失败 {keywords}: {e}")
            import traceback
            self.logger.error(f"🔍 [DEBUG] 详细错误堆栈: {traceback.format_exc()}")
            return []

    def get_macro_data(self) -> Dict[str, Any]:
        """获取宏观经济数据"""
        start_time = datetime.now()
        self.debug_logger.debug(f"获取宏观数据开始: timestamp={start_time.isoformat()}")

        try:
            # 如果ak可用，尝试获取真实数据
            if hasattr(self, 'ak') and self.ak is not None:
                try:
                    self.debug_logger.info("尝试从akshare获取真实宏观数据")
                    df = self.ak.index_zh_a_hist(
                        symbol="000300",
                        period="daily",
                        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                        end_date=datetime.now().strftime('%Y%m%d')
                    )

                    if not df.empty:
                        latest = df.iloc[-1]
                        result = {
                            "market_index": "沪深300",
                            "current_value": latest['收盘'],
                            "change_pct": ((latest['收盘'] - latest['开盘']) / latest['开盘']) * 100,
                            "volume": latest['成交量'],
                            "date": latest['日期'],
                            "data_source": "akshare"
                        }
                        end_time = datetime.now()
                        execution_time = (end_time - start_time).total_seconds()
                        self.logger.info(f"宏观数据获取成功: 沪深300, data_size=1, execution_time={execution_time:.2f}s, data_source=akshare")
                        return result
                except Exception as e:
                    self.logger.error(f"akshare获取宏观数据失败: {e}")
                    raise RuntimeError(f"无法获取真实宏观数据: {e}")
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.logger.info(f"宏观数据获取失败: 沪深300, execution_time={execution_time:.2f}s, error={str(e)}")
            self.logger.error(f"获取宏观数据失败: {e}")
            return {"market_index": "沪深300", "error": str(e)}

    def get_sector_data(self, sector: str) -> Dict[str, Any]:
        """获取行业数据"""
        start_time = datetime.now()
        self.debug_logger.debug(f"获取行业数据开始: sector={sector}, timestamp={start_time.isoformat()}")

        try:
            # 获取行业板块数据
            df = ak.stock_board_industry_name_em()
            if sector in df['板块名称'].values:
                sector_info = df[df['板块名称'] == sector].iloc[0]
                result = {
                    "sector_name": sector,
                    "change_pct": sector_info['涨跌幅'],
                    "turnover": sector_info['换手率'],
                    "leader_stocks": sector_info['领涨股票']
                }
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                self.logger.info(f"行业数据获取成功: sector={sector}, data_size=1, execution_time={execution_time:.2f}s, data_source=akshare")
                return result
            else:
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                self.logger.info(f"行业数据获取成功: sector={sector}, data_size=0, execution_time={execution_time:.2f}s, status=未找到该行业")
                return {"sector_name": sector, "error": "未找到该行业"}
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self.logger.info(f"行业数据获取失败: sector={sector}, execution_time={execution_time:.2f}s, error={str(e)}")
            self.logger.error(f"获取行业数据失败 {sector}: {e}")
            return {"sector_name": sector, "error": str(e)}

    def clear_cache(self):
        """清空缓存"""
        start_time = datetime.now()
        cache_size = len(self._cache)
        self._cache.clear()
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        self.debug_logger.debug(f"缓存清理完成: previous_cache_size={cache_size}, execution_time={execution_time}")
        self.logger.info("数据源缓存已清空")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            "cache_size": len(self._cache),
            "cache_ttl": self._cache_ttl,
            "cached_functions": list(set(str(key).split('_')[0] for key in self._cache.keys()))
        }
        self.debug_logger.debug(f"缓存统计信息: {stats}")
        return stats


# 全局数据源适配器实例
data_adapter = DataSourceAdapter()


def get_data_adapter() -> DataSourceAdapter:
    """获取全局数据源适配器实例"""
    return data_adapter


# CrewAI工具包装器
class DataRetrievalTool(BaseTool):
    """数据检索工具基类"""

    def __init__(self, adapter: DataSourceAdapter, name: str, description: str):
        # 使用属性避免被CrewAI BaseTool过滤
        self._adapter = adapter
        super().__init__(name=name, description=description)

    def get_adapter(self) -> DataSourceAdapter:
        """获取数据适配器"""
        # CrewAI可能过滤掉了_adapter属性，我们需要从其他地方获取
        if hasattr(self, '_adapter'):
            return self._adapter
        else:
            # 如果_adapter不存在，使用全局数据适配器
            from crewai_system.src.tools.data_sources import get_data_adapter
            return get_data_adapter()

    @property
    def logger(self):
        """获取日志记录器"""
        return get_logger("data")

    def _run(self, **kwargs) -> Any:
        """工具执行方法"""
        try:
            return self.execute(**kwargs)
        except Exception as e:
            self.logger.error(f"工具执行失败: {e}")
            return {"error": str(e)}

    def execute(self, **kwargs) -> Any:
        """具体的执行逻辑，子类必须实现"""
        raise NotImplementedError


class PriceHistoryTool(DataRetrievalTool):
    """价格历史获取工具"""

    def __init__(self, adapter: DataSourceAdapter):
        super().__init__(adapter, "Get Price History", "获取股票价格历史数据")

    def _run(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """执行价格历史获取"""
        self.logger.info(f"开始执行价格历史获取: {ticker} ({start_date} - {end_date})")
        result = self.get_adapter().get_price_history_safe(ticker, start_date, end_date)
        self.logger.info(f"价格历史获取完成: {ticker}, 数据条数: {len(result)}")
        return result


class FinancialMetricsTool(DataRetrievalTool):
    """财务指标获取工具"""

    def __init__(self, adapter: DataSourceAdapter):
        super().__init__(adapter, "Get Financial Metrics", "获取股票财务指标数据")

    def _run(self, ticker: str) -> Dict[str, Any]:
        """执行财务指标获取"""
        self.logger.info(f"开始执行财务指标获取: {ticker}")
        result = self.get_adapter().get_financial_metrics_safe(ticker)
        self.logger.info(f"财务指标获取完成: {ticker}, 指标数量: {len(result)}")
        return result


class MarketDataTool(DataRetrievalTool):
    """市场数据获取工具"""

    def __init__(self, adapter: DataSourceAdapter):
        super().__init__(adapter, "Get Market Data", "获取股票市场数据")

    def _run(self, ticker: str) -> Dict[str, Any]:
        """执行市场数据获取"""
        self.logger.info(f"开始执行市场数据获取: {ticker}")
        result = self.get_adapter().get_market_data_safe(ticker)
        self.logger.info(f"市场数据获取完成: {ticker}")
        return result


class NewsSearchTool(DataRetrievalTool):
    """新闻搜索工具"""

    def __init__(self, adapter: DataSourceAdapter):
        super().__init__(adapter, "Search Financial News", "搜索财经新闻")

    def _run(self, keywords: str, num_articles: int = 10) -> List[Dict[str, Any]]:
        """执行新闻搜索"""
        self.logger.info(f"开始执行新闻搜索: {keywords}, 数量: {num_articles}")
        result = self.get_adapter().search_financial_news_safe(keywords, num_articles)
        self.logger.info(f"新闻搜索完成: {keywords}, 搜索到 {len(result)} 条新闻")
        return result


class MacroDataTool(DataRetrievalTool):
    """宏观数据获取工具"""

    def __init__(self, adapter: DataSourceAdapter):
        super().__init__(adapter, "Get Macro Data", "获取宏观经济数据")

    def _run(self) -> Dict[str, Any]:
        """执行宏观数据获取"""
        self.logger.info("开始执行宏观数据获取")
        result = self.get_adapter().get_macro_data()
        self.logger.info(f"宏观数据获取完成: {result.get('market_index', '未知')}")
        return result