"""
真实数据源集成模块
基于原LangGraph系统的API实现
"""

import sys
import os
import akshare as ak
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import numpy as np
import json
import time

from crewai_system.src.utils.logging_config import get_logger

# 设置日志记录器
logger = get_logger("real_data_sources")

# 设置请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

class RealDataSources:
    """真实数据源类"""

    def __init__(self):
        self.logger = get_logger("real_data_sources")
        self._cache = {}
        self._cache_ttl = 300  # 5分钟缓存

        # 尝试导入原系统API
        self._import_original_apis()

    def _import_original_apis(self):
        """导入原系统API"""
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            from src.tools.api import (
                get_financial_metrics,
                get_financial_statements,
                get_market_data,
                get_price_history
            )
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

        # 不使用新闻爬虫，统一使用akshare获取新闻数据
        self.news_crawler = None
        self.logger.info("不使用新闻爬虫模块，统一使用akshare获取新闻数据")

    def get_price_history(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取价格历史数据"""
        cache_key = f"price_history_{ticker}_{start_date}_{end_date}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return pd.DataFrame(cached_result)

        try:
            # 优先使用原系统API
            if self.original_api and 'get_price_history' in self.original_api:
                self.logger.info(f"使用原系统API获取{ticker}价格历史")
                result = self.original_api['get_price_history'](ticker, start_date, end_date)
            else:
                # 使用akshare直接获取
                self.logger.info(f"使用akshare获取{ticker}价格历史")
                result = self._get_price_history_akshare(ticker, start_date, end_date)

            if result is not None and not result.empty:
                self._cache_result(cache_key, result.to_dict('records'))
                return result
            else:
                self.logger.error(f"获取{ticker}价格历史为空")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取{ticker}价格历史失败: {e}")
            return pd.DataFrame()

    def _get_price_history_akshare(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用akshare获取价格历史"""
        try:
            # 转换日期格式
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            # 获取历史行情数据
            df = ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start_dt.strftime('%Y%m%d'),
                end_date=end_dt.strftime('%Y%m%d'),
                adjust="qfq"
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # 重命名列以匹配标准格式
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change_amount",
                "换手率": "turnover"
            })

            # 确保日期列为datetime类型
            df["date"] = pd.to_datetime(df["date"])

            # 计算技术指标
            df = self._calculate_technical_indicators(df)

            return df

        except Exception as e:
            self.logger.error(f"akshare获取价格历史失败: {e}")
            return pd.DataFrame()

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            if df.empty or len(df) < 20:
                return df

            # 计算动量指标
            df["momentum_1m"] = df["close"].pct_change(periods=20)
            df["momentum_3m"] = df["close"].pct_change(periods=60)
            df["momentum_6m"] = df["close"].pct_change(periods=120)

            # 计算成交量动量
            df["volume_ma20"] = df["volume"].rolling(window=20).mean()
            df["volume_momentum"] = df["volume"] / df["volume_ma20"]

            # 计算波动率指标
            returns = df["close"].pct_change()
            df["historical_volatility"] = returns.rolling(window=20).std() * np.sqrt(252)

            return df

        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
            return df

    def get_financial_metrics(self, ticker: str) -> Dict[str, Any]:
        """获取财务指标"""
        cache_key = f"financial_metrics_{ticker}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        try:
            # 优先使用原系统API
            if self.original_api and 'get_financial_metrics' in self.original_api:
                self.logger.info(f"使用原系统API获取{ticker}财务指标")
                result_list = self.original_api['get_financial_metrics'](ticker)
                result = result_list[0] if result_list else {}
            else:
                # 使用akshare直接获取
                self.logger.info(f"使用akshare获取{ticker}财务指标")
                result = self._get_financial_metrics_akshare(ticker)

            if result:
                self._cache_result(cache_key, result)
                return result
            else:
                self.logger.error(f"获取{ticker}财务指标为空")
                return {}

        except Exception as e:
            self.logger.error(f"获取{ticker}财务指标失败: {e}")
            return {}

    def _get_financial_metrics_akshare(self, ticker: str) -> Dict[str, Any]:
        """使用akshare获取财务指标"""
        try:
            # 获取实时行情数据
            realtime_data = ak.stock_zh_a_spot_em()
            if realtime_data is None or realtime_data.empty:
                return {}

            stock_data = realtime_data[realtime_data['代码'] == ticker]
            if stock_data.empty:
                return {}

            stock_data = stock_data.iloc[0]

            # 获取财务指标数据
            current_year = datetime.now().year
            financial_data = ak.stock_financial_analysis_indicator(
                symbol=ticker, start_year=str(current_year-1))

            if financial_data is None or financial_data.empty:
                return {}

            # 按日期排序并获取最新的数据
            financial_data['日期'] = pd.to_datetime(financial_data['日期'])
            financial_data = financial_data.sort_values('日期', ascending=False)
            latest_financial = financial_data.iloc[0] if not financial_data.empty else pd.Series()

            # 获取利润表数据
            try:
                income_statement = ak.stock_financial_report_sina(
                    stock=f"sh{ticker}", symbol="利润表")
                latest_income = income_statement.iloc[0] if not income_statement.empty else pd.Series()
            except:
                latest_income = pd.Series()

            # 构建财务指标
            def convert_percentage(value: float) -> float:
                try:
                    return float(value) / 100.0 if value is not None else 0.0
                except:
                    return 0.0

            metrics = {
                # 市场数据
                "market_cap": float(stock_data.get("总市值", 0)),
                "float_market_cap": float(stock_data.get("流通市值", 0)),

                # 盈利能力指标
                "return_on_equity": convert_percentage(latest_financial.get("净资产收益率(%)", 0)),
                "net_margin": convert_percentage(latest_financial.get("销售净利率(%)", 0)),
                "operating_margin": convert_percentage(latest_financial.get("营业利润率(%)", 0)),

                # 增长指标
                "revenue_growth": convert_percentage(latest_financial.get("主营业务收入增长率(%)", 0)),
                "earnings_growth": convert_percentage(latest_financial.get("净利润增长率(%)", 0)),
                "book_value_growth": convert_percentage(latest_financial.get("净资产增长率(%)", 0)),

                # 财务健康指标
                "current_ratio": float(latest_financial.get("流动比率", 0)),
                "debt_to_equity": convert_percentage(latest_financial.get("资产负债率(%)", 0)),
                "free_cash_flow_per_share": float(latest_financial.get("每股经营性现金流(元)", 0)),
                "earnings_per_share": float(latest_financial.get("加权每股收益(元)", 0)),

                # 估值比率
                "pe_ratio": float(stock_data.get("市盈率-动态", 0)),
                "price_to_book": float(stock_data.get("市净率", 0)),
                "price_to_sales": float(stock_data.get("总市值", 0)) / float(latest_income.get("营业总收入", 1)) if float(latest_income.get("营业总收入", 0)) > 0 else 0,
            }

            return metrics

        except Exception as e:
            self.logger.error(f"akshare获取财务指标失败: {e}")
            return {}

    def get_market_data(self, ticker: str) -> Dict[str, Any]:
        """获取市场数据"""
        cache_key = f"market_data_{ticker}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        try:
            # 优先使用原系统API
            if self.original_api and 'get_market_data' in self.original_api:
                self.logger.info(f"使用原系统API获取{ticker}市场数据")
                result = self.original_api['get_market_data'](ticker)
            else:
                # 使用akshare直接获取
                self.logger.info(f"使用akshare获取{ticker}市场数据")
                result = self._get_market_data_akshare(ticker)

            if result:
                self._cache_result(cache_key, result)
                return result
            else:
                self.logger.error(f"获取{ticker}市场数据为空")
                return {}

        except Exception as e:
            self.logger.error(f"获取{ticker}市场数据失败: {e}")
            return {}

    def _get_market_data_akshare(self, ticker: str) -> Dict[str, Any]:
        """使用akshare获取市场数据"""
        try:
            # 获取实时行情数据
            realtime_data = ak.stock_zh_a_spot_em()
            if realtime_data is None or realtime_data.empty:
                return {}

            stock_data = realtime_data[realtime_data['代码'] == ticker]
            if stock_data.empty:
                return {}

            stock_data = stock_data.iloc[0]

            # 提取市场数据
            result = {
                "market_cap": float(stock_data.get("总市值", 0)),
                "volume": float(stock_data.get("成交量", 0)),
                "average_volume": float(stock_data.get("成交量", 0)),
                "fifty_two_week_high": float(stock_data.get("52周最高", 0)),
                "fifty_two_week_low": float(stock_data.get("52周最低", 0)),
                "current_price": float(stock_data.get("最新价", 0)),
                "price_change_percent": float(stock_data.get("涨跌幅", 0)),
            }

            return result

        except Exception as e:
            self.logger.error(f"akshare获取市场数据失败: {e}")
            return {}

    def get_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """获取财务报表"""
        cache_key = f"financial_statements_{ticker}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        try:
            # 优先使用原系统API
            if self.original_api and 'get_financial_statements' in self.original_api:
                self.logger.info(f"使用原系统API获取{ticker}财务报表")
                result = self.original_api['get_financial_statements'](ticker)
            else:
                # 使用akshare直接获取
                self.logger.info(f"使用akshare获取{ticker}财务报表")
                result = self._get_financial_statements_akshare(ticker)

            if result:
                self._cache_result(cache_key, result)
                return result
            else:
                self.logger.error(f"获取{ticker}财务报表为空")
                return {}

        except Exception as e:
            self.logger.error(f"获取{ticker}财务报表失败: {e}")
            return {}

    def _get_financial_statements_akshare(self, ticker: str) -> Dict[str, Any]:
        """使用akshare获取财务报表"""
        try:
            # 获取资产负债表数据
            balance_sheet = ak.stock_financial_report_sina(
                stock=f"sh{ticker}", symbol="资产负债表")
            latest_balance = balance_sheet.iloc[0] if not balance_sheet.empty else pd.Series()

            # 获取利润表数据
            income_statement = ak.stock_financial_report_sina(
                stock=f"sh{ticker}", symbol="利润表")
            latest_income = income_statement.iloc[0] if not income_statement.empty else pd.Series()

            # 获取现金流量表数据
            cash_flow = ak.stock_financial_report_sina(
                stock=f"sh{ticker}", symbol="现金流量表")
            latest_cash_flow = cash_flow.iloc[0] if not cash_flow.empty else pd.Series()

            # 构建财务报表数据
            result = {
                "income_statement": {
                    "revenue": float(latest_income.get("营业总收入", 0)),
                    "net_income": float(latest_income.get("净利润", 0)),
                    "operating_profit": float(latest_income.get("营业利润", 0)),
                },
                "balance_sheet": {
                    "total_assets": float(latest_balance.get("资产总计", 0)),
                    "total_liabilities": float(latest_balance.get("负债合计", 0)),
                    "shareholders_equity": float(latest_balance.get("所有者权益合计", 0)),
                },
                "cash_flow": {
                    "operating_cash_flow": float(latest_cash_flow.get("经营活动产生的现金流量净额", 0)),
                    "investing_cash_flow": float(latest_cash_flow.get("投资活动产生的现金流量净额", 0)),
                    "financing_cash_flow": float(latest_cash_flow.get("筹资活动产生的现金流量净额", 0)),
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"akshare获取财务报表失败: {e}")
            return {}

    def get_stock_news(self, ticker: str, max_news: int = 10) -> List[Dict[str, Any]]:
        """获取股票新闻"""
        cache_key = f"stock_news_{ticker}_{max_news}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        try:
            # 直接使用akshare获取新闻
            self.logger.info(f"使用akshare获取{ticker}新闻")
            result = self._get_stock_news_akshare(ticker, max_news)

            if result:
                self._cache_result(cache_key, result)
                return result
            else:
                self.logger.error(f"获取{ticker}新闻为空")
                return []

        except Exception as e:
            self.logger.error(f"获取{ticker}新闻失败: {e}")
            return []

    def _get_stock_news_akshare(self, ticker: str, max_news: int) -> List[Dict[str, Any]]:
        """使用akshare获取股票新闻"""
        try:
            # 获取新闻列表
            news_df = ak.stock_news_em(symbol=ticker)
            if news_df is None or len(news_df) == 0:
                return []

            # 处理新闻数据
            news_list = []
            for _, row in news_df.head(max_news).iterrows():
                try:
                    news_item = {
                        "title": row.get("新闻标题", "").strip(),
                        "content": row.get("新闻内容", row.get("新闻标题", "")).strip(),
                        "publish_time": row.get("发布时间", ""),
                        "source": row.get("文章来源", "").strip(),
                        "url": row.get("新闻链接", "").strip(),
                        "keyword": row.get("关键词", "").strip()
                    }
                    news_list.append(news_item)
                except Exception as e:
                    self.logger.warning(f"处理单条新闻时出错: {e}")
                    continue

            return news_list

        except Exception as e:
            self.logger.error(f"akshare获取新闻失败: {e}")
            return []

    def get_news_sentiment(self, news_list: List[Dict[str, Any]]) -> float:
        """分析新闻情感"""
        try:
            # 直接使用简单的情感分析
            self.logger.info("使用简单情感分析")
            return self._simple_sentiment_analysis(news_list)

        except Exception as e:
            self.logger.error(f"情感分析失败: {e}")
            return 0.0

    def _simple_sentiment_analysis(self, news_list: List[Dict[str, Any]]) -> float:
        """简单情感分析"""
        if not news_list:
            return 0.0

        try:
            # 简单的关键词情感分析
            positive_keywords = ["增长", "上涨", "盈利", "利好", "突破", "创新高", "增长", "提升"]
            negative_keywords = ["下跌", "亏损", "利空", "风险", "下跌", "下滑", "危机", "问题"]

            sentiment_score = 0.0
            for news in news_list:
                content = f"{news.get('title', '')} {news.get('content', '')}"
                content_lower = content.lower()

                positive_count = sum(1 for keyword in positive_keywords if keyword in content_lower)
                negative_count = sum(1 for keyword in negative_keywords if keyword in content_lower)

                sentiment_score += (positive_count - negative_count)

            # 归一化到[-1, 1]范围
            if sentiment_score > 0:
                return min(sentiment_score / len(news_list) / 5, 1.0)
            else:
                return max(sentiment_score / len(news_list) / 5, -1.0)

        except Exception as e:
            self.logger.error(f"简单情感分析失败: {e}")
            return 0.0


    # 缓存方法
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        if cache_key in self._cache:
            cached_time, result = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                self.logger.debug(f"使用缓存结果: {cache_key}")
                return result
            else:
                del self._cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """缓存结果"""
        self._cache[cache_key] = (time.time(), result)

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self.logger.info("真实数据源缓存已清空")


# 全局真实数据源实例
real_data_sources = RealDataSources()


def get_real_data_sources() -> RealDataSources:
    """获取全局真实数据源实例"""
    return real_data_sources