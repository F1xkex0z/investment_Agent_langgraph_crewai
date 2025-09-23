#!/usr/bin/env python3
"""
日志清理脚本
清理分散的旧日志文件，保留统一的新日志文件
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_old_logs():
    """清理旧的、分散的日志文件"""

    # 日志目录
    logs_dir = Path("logs")
    crewai_logs_dir = Path("crewai_system/src/logs")

    # 保留的统一日志文件
    unified_logs = {
        'investment_system.log',
        'api_calls.log',
        'agents.log',
        'data_processing.log',
        'performance.log',
        'errors.log',
        'debug.log'
    }

    # 需要删除的旧日志文件模式
    old_log_patterns = {
        # 主要系统中的旧日志
        'agent_state.log',
        'api.log',
        'fundamentals_agent.log',
        'llm_clients.log',
        'macro_analyst_agent.log',
        'macro_news_agent.log',
        'main_workflow.log',
        'market_data_agent.log',
        'news_crawler.log',
        'portfolio_management_agent.log',
        'sentiment_agent.log',
        'structured_terminal.log',
        'technical_analyst_agent.log',
        'valuation_agent.log',

        # CrewAI系统中的旧日志
        'system.log',
        'agent.log',
        'task.log',
        'error.log',
        'performance.log'
    }

    print("🧹 开始清理日志文件...")

    # 统计信息
    deleted_count = 0
    preserved_count = 0
    total_size_freed = 0

    # 清理主日志目录
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            if log_file.name in unified_logs:
                preserved_count += 1
                print(f"✅ 保留统一日志: {log_file.name}")
            else:
                try:
                    size = log_file.stat().st_size
                    log_file.unlink()
                    deleted_count += 1
                    total_size_freed += size
                    print(f"🗑️  删除旧日志: {log_file.name} ({size} bytes)")
                except Exception as e:
                    print(f"❌ 删除失败 {log_file.name}: {e}")

    # 清理CrewAI日志目录
    if crewai_logs_dir.exists():
        for log_file in crewai_logs_dir.glob("*.log"):
            try:
                size = log_file.stat().st_size
                log_file.unlink()
                deleted_count += 1
                total_size_freed += size
                print(f"🗑️  删除CrewAI日志: {log_file.name} ({size} bytes)")
            except Exception as e:
                print(f"❌ 删除失败 {log_file.name}: {e}")

    print("\n📊 清理统计:")
    print(f"   删除文件数: {deleted_count}")
    print(f"   保留文件数: {preserved_count}")
    print(f"   释放空间: {total_size_freed:,} bytes ({total_size_freed/1024/1024:.2f} MB)")

    print("\n📁 当前日志文件:")
    if logs_dir.exists():
        for log_file in sorted(logs_dir.glob("*.log")):
            size = log_file.stat().st_size
            print(f"   - {log_file.name}: {size:,} bytes")

    print(f"\n✅ 日志清理完成！")
    print(f"💡 现在系统使用统一的日志文件，所有日志将集中记录到上述文件中。")

def archive_old_logs(days_to_keep: int = 30):
    """归档旧的日志文件"""

    logs_dir = Path("logs")
    if not logs_dir.exists():
        return

    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    archive_dir = logs_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    print(f"\n📦 归档超过 {days_to_keep} 天的日志文件...")

    archived_count = 0
    for log_file in logs_dir.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_date.timestamp():
            try:
                # 创建归档文件名
                archive_name = f"{log_file.stem}_{log_file.stat().st_mtime:%Y%m%d}{log_file.suffix}"
                archive_path = archive_dir / archive_name

                # 移动文件到归档目录
                shutil.move(str(log_file), str(archive_path))
                archived_count += 1
                print(f"📦 归档: {log_file.name} -> {archive_name}")
            except Exception as e:
                print(f"❌ 归档失败 {log_file.name}: {e}")

    if archived_count > 0:
        print(f"✅ 归档完成，共处理 {archived_count} 个文件")
    else:
        print("ℹ️  没有需要归档的文件")

if __name__ == "__main__":
    print("🔧 A股投资分析系统 - 日志清理工具")
    print("=" * 50)

    # 清理旧日志
    cleanup_old_logs()

    # 归档旧日志
    archive_old_logs(days_to_keep=30)

    print("\n💡 提示:")
    print("   - 系统现在使用统一的日志配置")
    print("   - 所有日志将记录到 logs/ 目录下的少数几个文件中")
    print("   - 调试信息请查看 debug.log")
    print("   - 错误信息请查看 errors.log")
    print("   - 性能信息请查看 performance.log")
    print("   - API调用请查看 api_calls.log")