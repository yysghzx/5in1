# ============================================================================
# 增加申万行业筛选
# 增加资金监控不动qmt账户外来资金
# 增加deepseek概念映射
# 增加5%缓冲资金防止股价上涨买不到票
# 增加小白注释
# 减少log信息
# 增加redis交易函数
# 只有在放量破均价后，还出现“跌破昨收 / 深破均价 2% / 日内高点回撤 5%”之一，才触发“前日涨停烂板放量卖出”
# ============================================================================
# 打板策略 v2.2.0
# 基于之前版本的策略 完整重构买点模型
# 根据同花顺指标监测波段卖出增加量能监测
# 策略作者：Aric
# 策略类型：打板
# 热点概念数据源：同花顺
# 数据平台：聚宽（JoinQuant）
# 优化内容：
# 1）	解决 早盘竞价容易碰到瀑布杀问题：把之前9.26分买入时间点往后挪到9.33买入，在买入前根据内外比等一系列指标判断买入时机，如果不适合买入则往后延迟到9.40根据指标执行买入操作。除此之外如果有持仓则会在9.32分判断是否切换仓位买入当日优选个股。
# 2）	盘后内存回收提升策略代码执行性能，防止程序崩溃：15:30 清除 volume_data_cache 、 score_cache 等缓存并调用 gc.collect() ，防止OOM崩溃。
# 3）	优化业绩估值过滤功能：在每年4月和10月增加绩优股过滤，只有绩优股才会考虑买入。
# 4）	优化日志等若干问题：比如增加实盘执行效率日志跟踪

# ============================================================================
#
# 原始参考来源：
# - https://www.joinquant.com/post/65495  年化701%！五合一+热门概念加强版
# - https://www.joinquant.com/post/60627  打板策略实盘第一天收获涨停今日3连板
# - https://www.joinquant.com/post/59441  夏普43 打板五合一
# - https://www.joinquant.com/post/59300  打板策略五合一-临时版v5-加强版
# - https://www.joinquant.com/post/57458  五合一策略魔改 强到你不敢相信
# - https://www.joinquant.com/post/57372  打板策略五合一
# ============================================================================

from jqdata import *
from jqfactor import *
from jqlib.technical_analysis import *
import pandas as pd
import numpy as np
import time
import datetime as dt
from datetime import datetime, timedelta
import math
import gc

# 导入自定义redis交易函数
from send_to_redis import (
    order_zzy as order, 
    order_target_zzy as order_target, 
    order_value_zzy as order_value,
    order_target_value_zzy as order_target_value
)

# ============================================================================
# 引入同花顺热点概念公用函数
# ========================================================================
from test_hot_concept_utils import (
    set_g, set_logger, set_deepseek_api_key,
    get_all_hot_concepts_optimized,
    _get_hot_concepts_for_date,
    fetch_and_map_hot_concepts,
    convert_to_jq_code,
    ConceptMapper
)

"""
小白阅读指南
1. `initialize`
   策略启动入口。初始化全局变量 g，并注册所有定时任务。
2. `get_stock_list`
   每日选股入口。把股票分进不同模式的候选池。
3. `buy` / `execute_buy`
   `buy` 负责判断当前是否允许买，`execute_buy` 负责真正下单。
4. `sell_for_rebalance` / `sell_limit_per5min` / `sell2`
   卖出与风控逻辑，分别处理早盘调仓、盘中监控、尾盘止盈止损。

常见缩写
- `lb`: 连板龙头
- `yje`: 一进二
- `rzq`: 弱转强
- `dk`: 低开
- `fxsbdk`: 反向首板低开

这套策略的大致日内节奏
1. 09:25 `record_morning_stats`
   先看大盘环境，决定今天偏进攻还是偏防守。
2. 09:26~09:28 `get_stock_list`
   连续尝试选股，避免集合竞价数据晚到。
3. 09:32 `sell_for_rebalance`
   如果旧持仓开盘表现差，先卖掉腾仓位。
4. 09:33 `buy_after_auction_filter`
   对候选股做一次开盘后的内外比过滤，通过才买。
5. 09:40 `buy_after_auction_filter_retry`
   刚才没通过的票，再给一次机会。
6. 盘中 `sell_limit_per5min` / `sell2` / `execute_buy`
   每隔几分钟检查一次该不该卖，卖完后也可能补入新票。
7. 15:00 以后
   记录盘后统计、交易日志，并清理缓存。
"""


# ============================================================================
# 1. 初始化函数、日志函数
# ========================================================================
def initialize(context):
    """
    策略初始化入口。

    这里主要做三件事：
    1. 设置平台运行参数；
    2. 初始化全局状态 g；
    3. 注册全天要自动执行的任务。
    """
    set_option('use_real_price', True)
#     log.set_level('system', 'error')
    set_option('avoid_future_data', True)

    g.is_empty = False

    # ===== 账户与仓位配置 =====
    g.position_limit = 2  # 最大持仓数量，可配置
    g.cash_reserve_ratio = 0.05  # 保留5%资金缓冲，防止QMT执行时价格小幅上涨导致不够买

    # ===== 评分与排序配置 =====
    g.jqfactor = 'VOL5'  # 5日平均换手率（只是做为示例）
    g.sort = True  # 选取因子值最小

    # ===== 各策略模式的候选池 =====
    # 每天选股后，这些列表会被重新填充。
    g.emo_count = []
    g.gap_up = []
    g.gap_down = []
    g.reversal = []
    g.fxsbdk = []
    g.lblt = []
    g.hot_concepts_cache = []
    g.cache_max_days = 5
    g.min_score = 14
    g.qualified_stocks = []
    g.lblt_stocks = []
    g.rzq_stocks = []
    g.gk_stocks = []
    g.dk_stocks = []
    g.fxsbdk_stocks = []
    g.last_trade_info = None
    g.score_cache = {}  # 存储股票评分结果
    g.concept_num = 8  # 缓每日热点概念最大个数
    g.priority_config = []
    g.dynamic_params = {}  # 新增动态参数字典
    g.stocks_limit_up_today = set()
    g.max_sell_vol_ratio = 1.4  # 最大波段卖量比1.4
    g.morning_rebalance_vol_ratio = 2.1  # 早盘调仓量比阈值
    g.limit_break_sell_vol_ratio = 1.8  # 前日涨停烂板卖出的最低估算量比
    g.limit_break_sell_vwap_discount = 0.98  # 深破当日均价线才确认走弱
    g.limit_break_sell_intraday_retreat = 0.05  # 从日内高点回撤超过5%才确认走弱

    # 新增：选股完成标志
    g.stock_list_done = False
    g._deferred_buy_stocks = []  # 09:32内外比不通过的股票，推迟到09:40二次确认

    # 添加日志统计变量
    g.trade_stats = {
        'daily_returns': [],  # 每日收益
        'position_stats': {},  # 持仓统计
        'market_stats': {},  # 市场统计
        'trade_details': []  # 交易明细
    }

    # ===== 热点概念模块全局设置 =====
    from test_hot_concept_utils import set_g, set_logger, set_deepseek_api_key
    set_g(g)
#     set_logger(log)
    set_deepseek_api_key("sk-db2d0d33cf4747d5bae53ecb21de49ad")  # TODO: 替换为你的真实 DeepSeek API Key

    # ===== 定时任务总表 =====
    # 可以把 run_daily 理解成“到了这个时间，就自动调用对应函数”。
    run_daily(record_morning_stats, '09:25')  # 盘前数据统计
    run_daily(record_closing_stats, '15:00')  # 盘后数据统计
    # 原选股任务 (09:28:00) 替换为三次重试
    run_daily(get_stock_list, '09:26:00')  # 第一次尝试
    run_daily(get_stock_list, '09:27:00')  # 第二次尝试
    run_daily(get_stock_list, '09:28:00')  # 第三次尝试
    run_daily(buy, '14:51:00')  # 周五下午建仓
    run_daily(sell_limit_down, time='09:28', reference_security='000300.XSHG')
    run_daily(log_daily_trades, '15:05')  # 每日15:05记录当日交易
    run_daily(daily_garbage_collection, '15:30')  # 每日盘后清理内存碎片，防OOM

    # 新增：09:32早盘调仓仅卖出 + 内外比过滤买入 + 09:40二次确认
    run_daily(sell_for_rebalance, '09:32:00', reference_security='000300.XSHG')
    run_daily(buy_after_auction_filter, '09:33:00', reference_security='000300.XSHG')
    run_daily(buy_after_auction_filter_retry, '09:40:00', reference_security='000300.XSHG')

    # 优化：sell2函数调度（9:31~14:56每15分钟一次，避开竞价时间）
    sell2_times = [
        # 上午时段
        "10:31", "11:01",
        # 下午时段（跳过午间休市）
        "13:31", "14:01", "14:31", "14:50"  # 最后一次为14:50（替代15:00）
    ]
    for time_str in sell2_times:
        run_daily(sell2, time=time_str, reference_security='000300.XSHG')

    # 优化：使用循环设置每5分钟检测任务
    sell_per5min_times = []

    # 1. 上午时间段：9:35-9:59 每5分钟，10:01-10:26 每5分钟
    for hour in range(9, 11):
        start_minute = 35 if hour == 9 else 1  # 9点从35分开始，10点从1分开始
        end_minute = 60 if hour == 9 else 27  # 9点到59分，10点到26分结束
        for minute in range(start_minute, end_minute, 5):
            time_str = f"{hour:02d}:{minute:02d}"
            sell_per5min_times.append(time_str)

    # 2. 下午时间段：13:01-13:59 每5分钟，14:01-14:46 每5分钟
    for hour in range(13, 15):
        start_minute = 1 if hour == 13 else 1  # 13点/14点从1分开始
        end_minute = 60 if hour == 13 else 47  # 13点到59分，14点到46分结束
        for minute in range(start_minute, end_minute, 5):
            time_str = f"{hour:02d}:{minute:02d}"
            sell_per5min_times.append(time_str)

    # ========== 设置 sell_limit_per5min 定时任务 ==========
    for time_str in sell_per5min_times:
        run_daily(sell_limit_per5min, time=time_str, reference_security='000300.XSHG')

    # ========== 设置 execute_buy 定时任务（sell后1分钟），仅在周一到周四执行 ==========
    for time_str in sell_per5min_times:
        # 解析时间字符串为 datetime 对象
        base_time = dt.datetime.strptime(time_str, "%H:%M")
        # 加1分钟
        delay_time = base_time + timedelta(minutes=1)
        # 转回 HH:MM 格式字符串（处理进位，如 9:59+1=10:00）
        delay_time_str = delay_time.strftime("%H:%M")
        # 设置 execute_buy 定时任务
        run_daily(execute_buy, time=delay_time_str, reference_security='000300.XSHG')


# 根据市场环境更新策略优先级
def update_strategy_priority(trend):
    """根据市场趋势更新策略优先级"""
    # 根据分析结果设置不同市场环境下的策略优先级
    if trend == 'down':
        # 下跌市场: 反向首板低开(100%)、一进二(67%)、弱转强(0%)
        g.priority_config = ["lb", "fxsbdk", "yje", "rzq", "dk"]
    elif trend == 'strong_up':
        # 强势上涨市场: 连板龙头(56%)、弱转强(50%)、一进二(50%)
        g.priority_config = ["lb", "rzq", "yje", "fxsbdk", "dk"]
    elif trend == 'flat':
        # 平稳市场: 连板龙头(67%)、一进二(41%)
        g.priority_config = ["lb", "rzq", "yje", "fxsbdk", "dk"]
    elif trend == 'up':
        # 上涨市场: 一进二(40%)、连板龙头(35%)
        g.priority_config = ["yje", "lb", "rzq", "fxsbdk", "dk"]
    else:
        # 默认优先级
        g.priority_config = ["lb", "rzq", "yje", "dk", "fxsbdk"]
    # 存储当前策略优先级
    g.trade_stats['strategy_priority'] = {
        'trend': trend,
        'priority': g.priority_config
    }
    # log.info(f"策略优先级已更新: {trend} -> {' > '.join(g.priority_config)}")
#     log.info(f"根据市场趋势 [{trend}] 更新策略优先级: {' > '.join(g.priority_config)}")


def log_daily_trades(context):
    """
    记录每日交易日志
    """
    try:
        if not hasattr(g, 'today_trades'):
            log.info("今日无交易")
#             log.info("今日无交易")
            return

        # 统计交易情况
        total_trades = len(g.today_trades)
        if total_trades == 0:
            log.info("今日无交易")
#             log.info("今日无交易")
            return
        buy_trades = [trade for trade in g.today_trades if trade['action'] == '买入']
        sell_trades = [trade for trade in g.today_trades if trade['action'] == '卖出']

        # 计算总体盈亏
        total_profit_pct = sum(trade.get('profit_pct', 0) for trade in sell_trades) / len(
            sell_trades) if sell_trades else 0
        trade_summary = [f"{trade['stock']}{trade['action']}" for trade in g.today_trades]
        log.info(
            f"交易结果: 总计{total_trades}笔, 买入{len(buy_trades)}笔, 卖出{len(sell_trades)}笔, "
            f"平均盈亏{total_profit_pct:.2%}, 明细: {', '.join(trade_summary)}"
        )
#         log.info(
#             f"交易结果：总{total_trades}笔，买入{len(buy_trades)}笔，卖出{len(sell_trades)}笔，"
#             f"平均盈亏{total_profit_pct:.2%}，明细：{', '.join(trade_summary)}"
#         )

        # 重置今日交易记录
        g.today_trades = []
        # 重置全局变量
        g.gap_up = []
        g.gap_down = []
        g.reversal = []
        g.fxsbdk = []
        g.lblt = []
        g.lblt_stocks = []
        g.rzq_stocks = []
        g.gk_stocks = []
        g.dk_stocks = []
        g.stock_list_done = False
        return

    except Exception as e:
#         log.error(f"记录每日交易日志失败: {str(e)}")


# ==================== 内存管理 ====================

        pass
def daily_garbage_collection(context):
    """每日盘后强制清除缓存与垃圾回收，防止 OOM 崩溃"""
    if hasattr(g, 'volume_data_cache'):
        g.volume_data_cache.clear()
    if hasattr(g, 'score_cache'):
        g.score_cache.clear()
    if hasattr(g, 'high_risk_stocks_today'):
        g.high_risk_stocks_today = []
    gc.collect()


# ==================== 内外比工具函数 ====================

def _compute_inside_outside_ratio(stock, end_dt, tick_count=300):
    """
    基于 get_ticks 逐笔数据计算当日开盘后累计内外比。
    get_ticks 无"买/卖"方向字段，基于相邻tick的 current 变化推断：
      - current 较上笔上涨 → 外盘（主动买入推动价格）
      - current 较上笔下跌 → 内盘（主动卖出压低价格）
    返回 (外盘量, 内盘量, 内外比)，内外比 = 外盘/内盘。
    """
    try:
        ticks = get_ticks(stock, end_dt=end_dt, count=tick_count, df=True)
        if ticks is None or len(ticks) == 0:
            return 0, 0, 0.0

        buy_vol = 0.0
        sell_vol = 0.0
        prev_current = float(ticks['current'].iloc[0] or 0)

        for i in range(1, len(ticks)):
            cur = float(ticks['current'].iloc[i] or 0)
            vol = float(ticks['volume'].iloc[i] or 0)
            if cur > prev_current:
                buy_vol += vol
            elif cur < prev_current:
                sell_vol += vol
            prev_current = cur

        if sell_vol <= 0:
            return buy_vol, 0.0, 999.0 if buy_vol > 0 else 0.0

        ratio = round(buy_vol / sell_vol, 3)
        return round(buy_vol, 1), round(sell_vol, 1), ratio
    except Exception as e:
#         log.warning(f"{stock} 计算内外比失败: {e}")
        return 0, 0, 0.0


# ==================== 早盘仅卖出（不买入）====================

def sell_for_rebalance(context):
    """
    09:32执行：基于当日涨幅+量比判断是否卖出持仓腾仓位。
    条件1：当日涨幅<5% + 成本<5% + 1min量比>阈值 → 卖出
    条件2：当日涨幅>=5% + 1min量比>28 → 天量止盈
    """
    try:
        current_hour = context.current_dt.hour
        current_minute = context.current_dt.minute
        if current_hour < 9 or (current_hour == 9 and current_minute < 32):
            return
    except Exception:
        pass

    if not hasattr(g, 'qualified_stocks') or not g.qualified_stocks:
#         log.debug("【早盘调仓】候选池为空，无需腾仓位")
        return

    current_data = get_current_data()
    holding_stocks = list(context.portfolio.positions.keys())
    stock_profits = []

    for stock in holding_stocks:
        if stock not in current_data:
            continue
        position = context.portfolio.positions[stock]
        current_price = current_data[stock].last_price
        avg_cost = position.avg_cost
        cost_profit_pct = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0

        yesterday_close = None
        try:
            yc_hist = attribute_history(stock, 1, '1d', ['close'], skip_paused=True)
            if len(yc_hist) >= 1:
                yesterday_close = float(yc_hist['close'].iloc[-1])
        except Exception:
            pass
        day_gain_pct = (current_price - yesterday_close) / yesterday_close if (yesterday_close and yesterday_close > 0) else 0

        vol_ratio = get_1min_volume_ratio(stock, context, period=5)

        if day_gain_pct < 0.05:
            morning_threshold = getattr(g, 'morning_rebalance_vol_ratio', 2.1)
            if vol_ratio is not None and vol_ratio > morning_threshold and cost_profit_pct < 0.05:
                stock_profits.append((stock, day_gain_pct))
            else:
#                 log.debug(f"【早盘调仓】{stock} 日涨幅{day_gain_pct:.1%}<5% 量比{vol_ratio} 不满足卖出")
                pass
        elif day_gain_pct >= 0.05 and vol_ratio is not None and vol_ratio > 28:
            stock_profits.append((stock, day_gain_pct))
            log.info(f"早盘调仓-天量止盈: {stock}, 日涨幅{day_gain_pct:.1%}, 量比={vol_ratio:.2f}")
#             log.info(f"【早盘调仓-天量止盈】{stock} 日涨幅{day_gain_pct:.1%} 量比={vol_ratio:.2f}>28 → 卖出")

    if not stock_profits:
        return

    stock_profits.sort(key=lambda x: x[1])
    needed = min(len(stock_profits), g.position_limit)
    stocks_to_sell = [s for s, _ in stock_profits[:needed]]

    for stock in stocks_to_sell:
        log.info(f"早盘调仓-放量卖出: {stock}, 盈利不足5%且量能放大, 卖出腾仓")
        order_target_value(stock, 0)


# ==================== 09:33 内外比过滤买入 ====================

def buy_after_auction_filter(context, use_deferred=False):
    """
    内外比过滤买入。
    use_deferred=False (09:33)：对 g.qualified_stocks 全部候选计算内外比 → 通过则买入，拒绝暂存 g._deferred_buy_stocks
    use_deferred=True  (09:40)：对 g._deferred_buy_stocks 中被拒股票二次确认 → 通过则买入
    """
    if getattr(g, 'is_empty', False):
        return

    if use_deferred:
        tag = "二次确认"
        source = list(getattr(g, '_deferred_buy_stocks', []))
        tick_count = 500
        if not source:
#             log.debug(f"【内外比{tag}】无被拒股票，跳过")
            return
    else:
        tag = "过滤"
        if not hasattr(g, 'qualified_stocks') or not g.qualified_stocks:
#             log.info(f"【内外比{tag}】候选池为空，跳过")
            return
        source = list(g.qualified_stocks)
        tick_count = 300

    current_data = get_current_data()
    candidate_pool = [s for s in source
                      if s in current_data
                      and current_data[s].last_price < current_data[s].high_limit * 0.995
                      and s not in getattr(g, 'stocks_limit_up_today', set())
                      and s not in context.portfolio.positions]

    if not candidate_pool:
#         log.info(f"【内外比{tag}】候选全部涨停/已持仓/不在行情中，跳过")
        g._deferred_buy_stocks = []
        return

#     log.info(f"【内外比{tag}】开始计算 {len(candidate_pool)} 只候选股的内外比...")
    passed, rejected = [], []
    current_dt = context.current_dt

    for stock in candidate_pool:
        buy_vol, sell_vol, ratio = _compute_inside_outside_ratio(stock, end_dt=current_dt, tick_count=tick_count)
        if ratio >= 1.0 or (buy_vol == 0 and sell_vol == 0):
            passed.append(stock)
#             log.debug(f"  ✓ {stock} 外盘={buy_vol} 内盘={sell_vol} 内外比={ratio:.3f} → 通过")
        else:
            rejected.append(stock)
#             log.info(f"  ✗ {stock} 外盘={buy_vol} 内盘={sell_vol} 内外比={ratio:.3f} → 拒绝")

#     log.info(f"【内外比{tag}】通过{len(passed)}只 / 拒绝{len(rejected)}只")

    if not use_deferred:
        g._deferred_buy_stocks = list(rejected)

    if not passed:
#         log.info(f"【内外比{tag}】无候选通过，今日不买入" + ("" if use_deferred else "（被拒股票将09:40二次确认）"))
        return

    execute_buy(context, isFiltered=True, custom_stocks=passed)


def buy_after_auction_filter_retry(context):
    """09:40二次确认入口，委托到统一函数"""
    buy_after_auction_filter(context, use_deferred=True)


# ============================================================================
# 2. 概念筛选&缓存主函数、盘前数据统计函数
# ==========================================================================
def check_cache_status():
    """
    检查缓存状态
    """
    try:
        if not hasattr(g, 'hot_concepts_data_cache') or not g.hot_concepts_data_cache:
            return "缓存为空"

        cache_count = len(g.hot_concepts_data_cache)
        return f"正常({cache_count}个文件)"

    except Exception as e:
        return f"检查失败: {str(e)}"


def calculate_mainline_score_optimized(stock, context):
    """
    计算“主线概念”因子分。

    这项分数回答的是：
    “这只股票是不是踩在今天市场最热的题材线上？”

    计算方法很直白：
    - 先拿到今天的热门概念列表；
    - 再看这只股票自身有哪些概念；
    - 两边做匹配，命中几个热门概念就按数量加分。

    参数:
        stock: 股票代码
        context: 上下文对象
    返回:
        主线评分（匹配1个概念得2分，数量越多分数越高）
    """
    try:
        # 优先读缓存，避免每只股票都重复请求热门概念数据
        hot_concepts_result = getattr(g, 'hot_concepts_today', None) or get_all_hot_concepts_optimized(context)
        # 提取所有热门概念的名称列表
        hot_concepts_list = [concept['name'] for concept in hot_concepts_result['all_concepts']]
        hot_concepts_set = set(hot_concepts_list)  # 转为集合提高查询效率

        if not hot_concepts_set:
#             log.warning("热门概念列表为空，无法计算主线评分")
            return 0

        # 获取股票所属概念（提取概念名称，转为字符串列表）
        stock_info = get_security_info(stock)
        if not stock_info or not stock_info.concepts:
            return 0

        # 从概念字典中提取'name'字段，得到字符串列表
        stock_concepts = [concept['name'] for concept in stock_info.concepts]

        # 主线分按“命中的热门概念个数”来算，不重复计数
        matched_concepts = [c for c in stock_concepts if c in hot_concepts_set]
        unique_matched = list(set(matched_concepts))  # 去重处理
        match_count = len(unique_matched)  # 去重后的匹配数量

        # 按匹配数量计算分数（1个概念得2分，数量越多分数越高）
        mainline_score = match_count * 2 if match_count > 0 else 0

        return mainline_score

    except Exception as e:
#         log.error(f"计算主线评分时出错: {str(e)}")
        import traceback
#         log.error(traceback.format_exc())
        return 0


# ============================================================================
# 3-0. 行业趋势过滤（前置）
# ============================================================================

def get_industry_trend(stock, context, ma_short=5, ma_mid=20, ma_long=60):
    """
    通过申万二级行业成分股均值判断该股所在行业的趋势。
    过滤条件：MA5 > MA20（行业短期动能为正）。
    MA60 仅作日志参考，不参与过滤。
    返回: 'up' / 'down' / 'unknown'
    """
    try:
        # 1. 获取申万L2行业分类（粒度更细，约100个子行业）
        industry_info = get_industry(stock, date=context.previous_date)
        if not industry_info or stock not in industry_info:
            return 'unknown'

        sw_info = industry_info[stock].get('sw_l2')
        if not sw_info:
            return 'unknown'
        industry_code = sw_info.get('industry_code')
        industry_name = sw_info.get('industry_name', '')
        if not industry_code:
            return 'unknown'

        # 2. 获取申万L2行业成分股
        constituents = get_industry_stocks(industry_code, date=context.previous_date)
        if not constituents:
            return 'unknown'

        # 3. 剔除ST股和上市不足60日的新股，取全部有效成分股
        prev_date = context.previous_date
        valid_stocks = []
        for s in constituents:
            try:
                info = get_security_info(s)
                if info is None:
                    continue
                if 'ST' in info.display_name:
                    continue
                if (prev_date - info.start_date).days < 60:
                    continue
                valid_stocks.append(s)
            except Exception:
                continue

        if not valid_stocks:
            return 'unknown'

        # 4. 批量查询收盘价（一次 API 调用替代循环）
        needed = ma_mid + 5  # 25天，斜率只需11天，MA20日志用25足够
        price_arrays = []
        try:
            h = get_price(valid_stocks, end_date=prev_date, count=needed,
                          frequency='daily', fields=['close'], panel=False, fq='pre')
            if h is not None and not h.empty:
                for s in valid_stocks:
                    try:
                        prices = h[h['code'] == s]['close'].dropna().values
                        if len(prices) >= ma_mid and prices[0] > 0:
                            price_arrays.append(prices / prices[0])
                    except Exception:
                        continue
        except Exception:
            pass

        if not price_arrays:
            return 'unknown'

        # 5. 等权平均归一化序列，模拟行业等权指数（纯 Python，无 numpy/unstack 依赖）
        min_len = min(len(a) for a in price_arrays)
        if min_len < ma_mid:
            return 'unknown'
        aligned = [a[-min_len:] for a in price_arrays]
        normed = [sum(col) / len(col) for col in zip(*aligned)]

        ma5  = sum(normed[-ma_short:]) / ma_short
        ma20 = sum(normed[-ma_mid:])   / ma_mid

        # MA60（数据不足时为 None，仅用于日志和条件①）
        ma60 = None
        ma60_str = '-'
        if min_len >= ma_long:
            ma60 = sum(normed[-ma_long:]) / ma_long
            ma60_str = f'{ma60:.4f}'

        # 6. 纯斜率判断：只排除"持续下跌+无反转信号"的行业
        trend_reason = ''

        if min_len >= ma_short + 6:
            # 纯斜率判断：slope_recent = 近3日MA5变化量；slope_prev = 前3日MA5变化量
            slope_recent = (sum(normed[-ma_short:]) / ma_short
                            - sum(normed[-(ma_short+3):-3]) / ma_short)
            slope_prev   = (sum(normed[-(ma_short+3):-3]) / ma_short
                            - sum(normed[-(ma_short+6):-6]) / ma_short)
            if slope_recent > 0:
                # MA5 本身在上升
                trend = 'up'
                trend_reason = f'MA5上升({slope_recent:+.4f})'
            elif slope_recent > slope_prev:
                # MA5 仍下降但斜率在改善（下跌减速/底背离）
                trend = 'momentum'
                trend_reason = f'斜率衰减({slope_prev:+.4f}→{slope_recent:+.4f})'
            else:
                # 下跌加速
                trend = 'down'
                trend_reason = f'斜率加速({slope_prev:+.4f}→{slope_recent:+.4f})'
        else:
            trend = 'down'
            trend_reason = '数据不足'

        stock_name = stock
        try:
            stock_info = get_security_info(stock)
            if stock_info and getattr(stock_info, 'display_name', None):
                stock_name = stock_info.display_name
        except Exception:
            pass

        log.debug(f"  行业趋势(L2) {stock}({stock_name}) [{industry_name}({industry_code})] "
                  f"样本{len(price_arrays)}只 "
                  f"MA{ma_short}={ma5:.4f} MA{ma_mid}={ma20:.4f} MA{ma_long}={ma60_str}"
                  f" [{trend_reason}] → {trend}")
        return trend

    except Exception as e:
        log.warning(f"  get_industry_trend({stock}) 失败: {e}")
        return 'unknown'


def filter_stocks_by_industry_trend(stocks, context):
    """
    前置行业趋势过滤：只保留申万L2行业 MA5>MA20 或近5日涨幅>3% 的股票。
    行业趋势缓存到 g._industry_trend_cache，同一行业当日只查询一次。
    返回 (passed, filtered_out) 两个列表。
    """
    if not hasattr(g, '_industry_trend_cache'):
        g._industry_trend_cache = {}

    passed, filtered_out = [], []

    for stock in stocks:
        try:
            # 先查缓存（同行业多只股票只算一次）
            industry_info = get_industry(stock, date=context.previous_date)
            sw_info = (industry_info or {}).get(stock, {}).get('sw_l2', {})
            industry_code = sw_info.get('industry_code', '') if sw_info else ''

            if industry_code and industry_code in g._industry_trend_cache:
                trend = g._industry_trend_cache[industry_code]
            else:
                trend = get_industry_trend(stock, context)
                if industry_code:
                    g._industry_trend_cache[industry_code] = trend

            if trend == 'up':
                passed.append(stock)
            elif trend == 'momentum':
                # 价格>MA20 或斜率衰减/底背离，下跌动能耗尽，视为反转信号放行
                passed.append(stock)
#                 log.info(f"  ~ {stock} 行业反转信号，放行")
            elif trend == 'unknown':
                # 无法判断时放行，不误杀
                passed.append(stock)
#                 log.debug(f"  {stock} 行业趋势未知，放行")
            else:
                filtered_out.append(stock)
#                 log.info(f"  ✗ {stock} 行业趋势={trend}，过滤")

        except Exception as e:
#             log.warning(f"  filter_stocks_by_industry_trend({stock}) 异常: {e}，放行")
            passed.append(stock)

    return passed, filtered_out


# ============================================================================
# 3. 股票筛选主函数（修复版）
# ============================================================================
def filter_stocks_by_score_optimized(stocks, context, min_score=14, max_stocks=100):
    """
    评分筛选主函数。

    这一步可以理解为“细筛”：
    1. 先给每只股票算 6 个因子分；
    2. 再按不同模式追加专属过滤条件；
    3. 把通过的股票放进 g.qualified_stocks 供买入环节使用。

    六个因子分别是：
    - factor1_涨停: 最近涨停强度
    - factor2_技术: 均线、RSI、位置等技术形态
    - factor3_放量MA: 放量情况 + 均线关系
    - factor4_主线: 是否贴近热门概念主线
    - factor5_情绪: 大盘与个股相对情绪
    - factor6_主力资金: 资金净流入质量

    注意：
    总分高不代表一定入选，因为不同模式还有单独门槛，比如量比、主线分、资金分。
    """
    try:
        # log.info("=" * 60)
        # log.info(f"开始股票评分筛选，候选股票: {len(stocks)} 只，最低分数: {min_score}")
        # log.info("=" * 60)

        # 获取各模式股票列表
        lblt_stocks = getattr(g, 'lblt_stocks', [])  # 连板龙头模式
        gk_stocks = getattr(g, 'gk_stocks', [])  # 一进二模式
        rzq_stocks = getattr(g, 'rzq_stocks', [])  # 弱转强模式
        dk_stocks = getattr(g, 'dk_stocks', [])  # 低开股票
        fxsbdk_stocks = getattr(g, 'fxsbdk_stocks', [])  # 放巡散步低开股票

        # 获取参数
        trend = g.dynamic_params.get('trend', 'flat')
        min_money_flow = 6
        min_total_score_yijin = 24
        rzq_min_vol = 0.7
        rzq_max_vol = 10.0
        rzq_max_total = 28
        dk_min_technical = 4
        dk_min_volume_ma = 2

        # 清空之前的评分缓存
        g.score_cache = {}

        qualified_stocks = []  # 只存储股票代码字符串
        score_records = []
        processing_stats = {
            'total_stocks': len(stocks),
            'processed_stocks': 0,
            'qualified_stocks': 0,
            'failed_stocks': 0,
            'high_score_stocks': 0,
            'medium_score_stocks': 0,
            'low_score_stocks': 0,
            'zero_score_stocks': 0,
            'processing_time': 0,
            'cache_status': 'unknown',
            'filtered_by_mainline': 0,
            'filtered_by_money_flow': 0,
            'filtered_by_volume_ratio': 0,
            'filtered_by_volume_ratio_weak_to_strong': 0,
            'filtered_by_volume_ratio_lblt': 0,
            'filtered_by_volume_ratio_first_to_second': 0,
            'filtered_by_total_score': 0,  # 新增此行
        }

        start_time = time.time()

        # 限制处理数量
        limited_stocks = stocks[:max_stocks] if len(stocks) > max_stocks else stocks
        if len(stocks) > max_stocks:
            # log.info(f"⚠️  股票数量过多，限制处理前 {max_stocks} 只股票")

        # 检查缓存状态
            pass
        cache_status = check_cache_status()
        processing_stats['cache_status'] = cache_status
        # log.info(f"热门概念缓存状态: {cache_status}")

        # 预先获取热门概念（全局缓存，避免每只股票重复调用）
        g.hot_concepts_today = get_all_hot_concepts_optimized(context)
        # log.info(f"热门概念预加载完成，共 {len(g.hot_concepts_today.get('all_concepts', []))} 个概念")

        # 预先获取大盘数据（情绪评分缓存，避免每只股票重复调用）
        try:
            g.index_data_today = attribute_history('000001.XSHG', 5, '1d',
                                                   ['close', 'volume'], skip_paused=True)
        except Exception as e:
            # log.warning(f"预加载大盘数据失败: {e}")
            g.index_data_today = None

        # 批量获取股票基本信息
        current_data = get_current_data()

        # ===== 前置过滤：行业趋势 =====
        g._industry_trend_cache = {}  # 每次选股重置行业缓存
        industry_passed, industry_filtered = filter_stocks_by_industry_trend(limited_stocks, context)
#         log.info(f"行业趋势过滤: {len(limited_stocks)} → {len(industry_passed)} 只通过，"
#                  f"{len(industry_filtered)} 只因行业下降/震荡被过滤")
        if industry_filtered:
            filtered_names = []
            for s in industry_filtered[:10]:
                try:
                    filtered_names.append(get_current_data()[s].name if s in get_current_data() else s)
                except Exception:
                    filtered_names.append(s)
#             log.info(f"  被过滤股票(前10): {filtered_names}")
        limited_stocks = industry_passed
        processing_stats['total_stocks'] = len(limited_stocks)
        # ===== 前置过滤结束 =====

        # 预先获取所有股票的资金流向数据
        money_flow_map = get_money_flow_map(context, limited_stocks)

        for i, stock in enumerate(limited_stocks):
            try:
                # 进度显示
                if i % 20 == 0 and i > 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_stock = elapsed_time / i
                    remaining_time = avg_time_per_stock * (len(limited_stocks) - i)

                # 判断股票模式类型
                is_lblt_stock = stock in lblt_stocks
                is_first_to_second_stock = stock in gk_stocks
                is_weak_to_strong_stock = stock in rzq_stocks
                is_dk_stock = stock in dk_stocks
                is_fxsbdk_stock = stock in fxsbdk_stocks

                # 确定股票模式
                if is_lblt_stock:
                    stock_mode = "连板龙头"
                elif is_first_to_second_stock:
                    stock_mode = "一进二"
                elif is_weak_to_strong_stock:
                    stock_mode = "弱转强"
                elif is_dk_stock:
                    stock_mode = "低开"
                elif is_fxsbdk_stock:
                    stock_mode = "放巡散步低开"
                else:
                    stock_mode = "未分类"  # 默认为未分类模式

                if stock_mode == "未分类":
                    # log.info(f"⛔ {stock} 无特定模式，过滤掉")
                    continue

                # 1. 获取基础评分结果（包含全部6个因子，主力资金因子已统一计算）
                score_result = calculate_buy_score_optimized(stock, context, money_flow_map)
                processing_stats['processed_stocks'] += 1

                if not score_result:
                    processing_stats['failed_stocks'] += 1
                    continue

                # 2. 提取各因子得分（主力资金因子使用统一计算结果）
                factor1 = score_result.get('factor1_涨停', 0)
                factor2 = score_result.get('factor2_技术', 0)
                factor3 = score_result.get('factor3_放量MA', 0)
                factor4 = score_result.get('factor4_主线', 0)
                factor5 = score_result.get('factor5_情绪', 0)
                factor6 = score_result.get('factor6_主力资金', 0)  # 统一使用calculate_main_force_flow_score结果
                total_score = factor1 + factor2 + factor3 + factor4 + factor5 + factor6

                # 3. 将评分结果缓存到全局变量（包含6个因子）
                g.score_cache[stock] = {
                    'total_score': total_score,
                    'factor1_涨停': factor1,
                    'factor2_技术': factor2,
                    'factor3_放量MA': factor3,
                    'factor4_主线': factor4,
                    'factor5_情绪': factor5,
                    'factor6_主力资金': factor6,  # 统一缓存结果
                    'timestamp': context.current_dt,
                    'is_lblt': is_lblt_stock,  # 标记是否为连板龙头
                    'stock_mode': stock_mode  # 记录股票模式
                }

                # 4. 统计评分分布
                if total_score >= 20:
                    processing_stats['high_score_stocks'] += 1
                elif total_score >= 15:
                    processing_stats['medium_score_stocks'] += 1
                elif total_score >= 10:
                    processing_stats['low_score_stocks'] += 1
                else:
                    processing_stats['zero_score_stocks'] += 1

                # 5. 第一层判断：先看“总分是否达到基础门槛”
                is_qualified = total_score >= min_score

                # 第二层判断：再看“模式专属规则”
                # 同样是高分股，如果不符合本模式特征，也会被过滤掉。
                filtered_reason = None
                if is_lblt_stock:
                    # 连板龙头：
                    # 这类票最看重情绪和主线，如果又没主线、情绪又弱，就不做。
                    if factor4 == 0 and factor5 < 12:
                        is_qualified = False
                        processing_stats['filtered_by_mainline'] += 1
                        filtered_reason = '连板龙头主线分为0且情绪分<12'
                    # 如果资金没有明显支持，同时涨停强度也一般，也过滤。
                    elif factor6 == 0 and factor1 < 2:
                        is_qualified = False
                        processing_stats['filtered_by_money_flow'] += 1
                        filtered_reason = '连板龙头主力资金为0且涨停分小于2'

                if is_first_to_second_stock:
                    # 一进二：
                    # 既要有足够资金支持，也不能热得太夸张，否则容易变成接力末端。
                    if factor4 > 18:
                        is_qualified = False
                        processing_stats['filtered_by_mainline'] += 1
                        filtered_reason = '一进二主线分>18'
                    if factor6 < min_money_flow:
                        is_qualified = False
                        processing_stats['filtered_by_money_flow'] += 1
                        filtered_reason = f'一进二主力资金<{min_money_flow}'
                    if total_score < min_total_score_yijin:
                        is_qualified = False
                        processing_stats['filtered_by_total_score'] += 1
                        filtered_reason = f'一进二总分<{min_total_score_yijin}'
                    # 量比可以理解为“当前活跃度”，一进二对量能比较敏感
                    volume_ratio = g.score_cache[stock].get('volume_ratio', None)
                    if volume_ratio is not None:
                        if total_score < 28 and volume_ratio < 2.6:
                            is_qualified = False
                            processing_stats['filtered_by_volume_ratio'] = processing_stats.get(
                                'filtered_by_volume_ratio', 0) + 1
                            filtered_reason = f'一进二总分<28且量比<2.6 (总分{total_score},量比{volume_ratio:.2f})'

                if is_dk_stock:
                    # 首板低开：
                    # 对技术形态和量价关系要求更直接，分不够就不过。
                    if factor2 < dk_min_technical:
                        is_qualified = False
                        processing_stats['filtered_by_technical'] = processing_stats.get('filtered_by_technical', 0) + 1
                        filtered_reason = f'首板低开技术分<{dk_min_technical}'
                    elif factor3 < dk_min_volume_ma:
                        is_qualified = False
                        processing_stats['filtered_by_volume_ma'] = processing_stats.get('filtered_by_volume_ma', 0) + 1
                        filtered_reason = f'首板低开放量MA分<{dk_min_volume_ma}'

                # 第三层判断：按模式检查量比范围
                # 量比太低表示不活跃，太高又可能是情绪过热或出货。
                try:
                    # 获取股票的量比数据
                    last_volume, last_2_volume, volume_ratio = get_volume_data(stock, context)

                    # log.info(f"获取股票 {stock} 的量比数据: {volume_ratio}, is_lblt_stock:{is_lblt_stock}")
                    # 根据不同模式设置量比范围限制
                    if is_lblt_stock:
                        # 连板龙头模式量能比范围：1.0~10.5（策略2范围）
                        if volume_ratio < 1.0 or volume_ratio > 10.5:
                            is_qualified = False
                            processing_stats['filtered_by_volume_ratio'] += 1
                            processing_stats['filtered_by_volume_ratio_lblt'] += 1
                            filtered_reason = f'连板龙头模式量比不符({volume_ratio:.2f}不在1.0~10.5范围内)'

                    elif is_weak_to_strong_stock or is_dk_stock or is_fxsbdk_stock:
                        # 弱转强模式个股量能比范围 0.7~4.2（策略2范围）
                        if volume_ratio < rzq_min_vol or volume_ratio > rzq_max_vol:
                            is_qualified = False
                            processing_stats['filtered_by_volume_ratio'] += 1
                            processing_stats['filtered_by_volume_ratio_weak_to_strong'] += 1
                            filtered_reason = f'弱转强模式量比不符({volume_ratio:.2f}不在{rzq_min_vol}~{rzq_max_vol}范围内)'

                    else:
                        # 未分类模式，使用更宽松的量比范围 0.4~10.5（策略2范围）
                        if volume_ratio < 0.4 or volume_ratio > 10.5:
                            is_qualified = False
                            processing_stats['filtered_by_volume_ratio'] += 1
                            filtered_reason = f'未分类模式量比不符({volume_ratio:.2f}不在0.4~10.5范围内)'

                    # 将量比信息添加到评分缓存中
                    g.score_cache[stock]['volume_ratio'] = volume_ratio

                except Exception as ve:
                    # log.warning(f"获取股票 {stock} 的量比数据失败: {str(ve)}")

                # 第四层判断：按模式再看“总分区间”
                # 有些模式不是分越高越好，而是要求落在特定区间里。
                    pass
                if is_qualified:
                    # 一进二模式总分范围：20~60分（策略2范围）
                    if is_first_to_second_stock and (total_score < 20 or total_score > 60):
                        is_qualified = False
                        filtered_reason = f'一进二模式总分不符({total_score}不在20~60范围内)'

                    # 弱转强模式总分要求：>=24分（策略2要求）
                    elif (is_weak_to_strong_stock or is_dk_stock or is_fxsbdk_stock):
                        if total_score < 24:
                            is_qualified = False
                            filtered_reason = f'弱转强模式总分<24'
                        elif is_weak_to_strong_stock and total_score > rzq_max_total:  # 仅对弱转强应用上限
                            is_qualified = False
                            filtered_reason = f'弱转强模式总分>{rzq_max_total}'

                if is_qualified:
                    qualified_stocks.append(stock)
                    processing_stats['qualified_stocks'] += 1

                # 6. 获取股票名称
                try:
                    if stock in current_data:
                        stock_name = current_data[stock].name
                    else:
                        security_info = get_security_info(stock)
                        stock_name = security_info.display_name if security_info else "未知"
                except:
                    stock_name = "未知"

                # 7. 保存日志用记录。
                # 后面打印结果、排查被过滤原因，都依赖这份结构化记录。
                record = {
                    '股票代码': stock,
                    '股票名称': stock_name,
                    '总评分': total_score,
                    '是否选中': '✓' if is_qualified else '✗',
                    'factor1_涨停': factor1,
                    'factor2_技术': factor2,
                    'factor3_放量MA': factor3,
                    'factor4_主线': factor4,
                    'factor5_情绪': factor5,
                    'factor6_主力资金': factor6,  # 统一记录结果
                    '是否连板龙头': '✓' if is_lblt_stock else '✗',
                    '被过滤原因': filtered_reason,
                    '量比': g.score_cache[stock].get('volume_ratio', '未知'),  # 添加量比信息到记录中
                    '模式': stock_mode  # 添加模式信息
                }
                score_records.append(record)

                # 8. 输出符合条件股票的详细信息（含6个因子）
            except Exception as e:
                processing_stats['failed_stocks'] += 1
                # log.error(f"处理股票 {stock} 时出错: {str(e)}")
                continue

        # 计算处理时间
        processing_stats['processing_time'] = time.time() - start_time
        log.info(f"股票筛选完成: {len(qualified_stocks)}只")

#         log.info(f"股票筛选完成：{len(qualified_stocks)}只")

        # 最终排序：
        # 先比总分，再比放量MA，最后比主力资金。
        score_records_sorted = sorted(score_records, 
                                     key=lambda x: (x['总评分'], x.get('factor3_放量MA', 0), x.get('factor6_主力资金', 0)), 
                                     reverse=True)
        score_records_limited = score_records_sorted

        # 更新合格股票列表为排序后并限制数量的结果
        qualified_stocks = [record['股票代码'] for record in score_records_limited if record['是否选中'] == '✓']

        # ========== 新增：限制股票数量不超过最大持仓数 ==========
        position_limit = getattr(g, 'position_limit', 2)
        if len(qualified_stocks) > position_limit:
            qualified_stocks = qualified_stocks[:position_limit]
            log.info(f"根据最大持仓数{position_limit}，限制股票数量为前{position_limit}只")
#             log.info(f"根据最大持仓数{position_limit}，限制股票数量为前{position_limit}只")

        # 显示前5只符合条件的股票详情
        if qualified_stocks:
            # log.info("🎯 符合条件的股票列表:")
            for i, stock in enumerate(qualified_stocks[:5]):  # 只显示前5只
                matching_record = next((r for r in score_records if r['股票代码'] == stock), None)
                if matching_record:
                    mode_tag = f"[{matching_record['模式']}]" if matching_record['模式'] != "未分类" else ""
                    volume_ratio_info = f"量比:{matching_record['量比']:.2f}" if isinstance(matching_record['量比'],
                                                                                            (int, float)) else "量比:未知"
                    # log.info(f"  {i + 1}. {stock} ({matching_record['股票名称']}){mode_tag} - "
                    #          f"总分: {matching_record['总评分']} "
                    #          f"[涨停:{matching_record['factor1_涨停']} "
                    #          f"技术:{matching_record['factor2_技术']} "
                    #          f"放量MA:{matching_record['factor3_放量MA']} "
                    #          f"主线:{matching_record['factor4_主线']} "
                    #          f"情绪:{matching_record['factor5_情绪']} "
                    #          f"主力资金:{matching_record['factor6_主力资金']} "
                    #          f"{volume_ratio_info}]")

            if len(qualified_stocks) > 5:
#                 log.info(f"  ... 共 {len(qualified_stocks)} 只符合条件股票")

                pass
        return qualified_stocks

    except Exception as e:
#         log.error(f"股票筛选过程出错: {str(e)}")
        import traceback
#         log.error(traceback.format_exc())
        return []


def get_money_flow_map(context, qualified_stocks):
    """
    获取合格股票的主力资金流数据并构建映射字典（包含所有日期记录）
    参数:
        context: 上下文对象
        qualified_stocks: 合格股票列表
    返回:
        money_flow_map: 主力资金流映射字典，键为股票代码，值为包含所有日期记录的列表
    """
    money_flow_map = {}
    try:
        if not qualified_stocks:
            return money_flow_map
            log.info("没有符合条件的股票")
            log.info("选股结果：0只，无")
            log.info("选股结果：0只，无")
            log.info("没有符合条件的股票")
#             log.info("合格股票列表为空，无需获取主力资金数据")
            return money_flow_map

        # 获取最近5个交易日的主力资金数据（保留所有日期，不做数量限制）
        end_date = context.previous_date
        trade_days = get_trade_days(end_date=end_date, count=5)
#         log.info(f"~~~日期范围: {trade_days[0]} 至 {end_date}, trade_days:{trade_days}")
        money_flow_df = get_money_flow(qualified_stocks, start_date=trade_days[0], end_date=end_date)
        if money_flow_df.empty:
#             log.info("未获取到主力资金流数据")
            return money_flow_map
        # 确定日期列（兼容'date'或'trade_date'字段）
        date_column = 'date' if 'date' in money_flow_df.columns else 'trade_date'
        if date_column not in money_flow_df.columns:
            # log.warning("资金流数据中未找到日期列，使用end_date作为默认日期")
        # 构建资金流映射字典（按股票代码聚合所有日期记录）
            pass
        for _, row in money_flow_df.iterrows():
            stock_code = row['sec_code']
            # 初始化股票对应的记录列表（首次出现时）
            if stock_code not in money_flow_map:
                money_flow_map[stock_code] = []
            # 优先使用数据中的日期，无则用end_date
            record_date = row[date_column] if date_column in money_flow_df.columns else end_date
            # 构建单条日期记录并添加到列表
            daily_record = {
                'date': record_date,
                'net_amount_main': row['net_amount_main'],  # 主力净额(万)
                'net_pct_main': row['net_pct_main'],  # 主力净占比(%)
                'net_amount_l': row['net_amount_l']
            }
            money_flow_map[stock_code].append(daily_record)
        # 输出每个股票的记录数量日志
        # for stock, records in money_flow_map.items():
        #     log.info(f"个股主力资金数据：{stock} 共 {len(records)} 条记录，日期范围: {records[0]['date']} 至 {records[-1]['date']}")
    except Exception as e:
        # log.warning(f"批量获取主力资金数据失败: {str(e)}")
        pass
    return money_flow_map


# 同花顺指标转换为Python函数
def calculate_ths_indicators(stock, context, period=60, unit='1d', log_debug=False):
    """
    计算技术指标，返回买卖信号、趋势颜色及最近信号信息。
    主信号（无未来函数，等待转折点确认后才产生信号）：
        - 波段买: ZIG转折点确认后（至少滞后1-2根K线）
        - 波段卖: ZIG转折点确认后 + 连续下跌形态
    返回字典包含：
        - buy_signals: 列表，信号字符串
        - sell_signals: 列表，信号字符串
        - trend_color: 'red' 表示上升通道，'green' 表示下降通道
        - last_signal_type: 最近一次信号的类型，若没有则为 None
        - last_signal_offset: 最近一次信号距离最后一根K线的偏移量
    """
    try:
        hist_data = attribute_history(stock, period, unit,
                                      ['open', 'close', 'high', 'low', 'volume'],
                                      skip_paused=True)

        if log_debug:
            # log.debug(f"{stock} 历史数据长度: {len(hist_data)}")

            pass
        if hist_data.empty or len(hist_data) < 20:
            if log_debug:
#                 log.debug(f"{stock} 数据不足20根，跳过")
                pass
            return {
                'buy_signals': [],
                'sell_signals': [],
                'trend_color': 'green',
                'last_signal_type': None,
                'last_signal_offset': None
            }

        # 提取OHLCV数据
        O = hist_data['open'].values
        C = hist_data['close'].values
        H = hist_data['high'].values
        L = hist_data['low'].values
        V = hist_data['volume'].values

        # ========== ZIG指标计算（无未来函数版本） ==========
        def calculate_zig(prices, threshold_pct=10):
            """
            ZIG指标（转折点确认版）：只计算已确认的历史转折点
            关键改进：不对最后一根K线进行转折点判定
            """
            if len(prices) < 3:
                return prices

            zig = np.zeros(len(prices))
            zig[0] = prices[0]

            last_extreme = prices[0]
            last_extreme_idx = 0
            trend = 0  # 0:未定义, 1:上升趋势, -1:下降趋势

            # 只计算到倒数第2根K线，最后一根K线不参与转折点判定
            max_calc_idx = len(prices) - 2

            for i in range(1, max_calc_idx + 1):
                if i == 1:
                    zig[i] = prices[i]
                    continue

                change_pct = abs(prices[i] - last_extreme) / last_extreme * 100 if last_extreme > 0 else 0

                if change_pct >= threshold_pct:
                    if prices[i] > last_extreme:
                        if trend != 1:
                            trend = 1
                            for j in range(last_extreme_idx + 1, i + 1):
                                t = (j - last_extreme_idx) / (i - last_extreme_idx)
                                zig[j] = last_extreme + (prices[i] - last_extreme) * t
                            last_extreme = prices[i]
                            last_extreme_idx = i
                        else:
                            last_extreme = prices[i]
                            last_extreme_idx = i
                            zig[i] = prices[i]
                    else:
                        if trend != -1:
                            trend = -1
                            for j in range(last_extreme_idx + 1, i + 1):
                                t = (j - last_extreme_idx) / (i - last_extreme_idx)
                                zig[j] = last_extreme + (prices[i] - last_extreme) * t
                            last_extreme = prices[i]
                            last_extreme_idx = i
                        else:
                            last_extreme = prices[i]
                            last_extreme_idx = i
                            zig[i] = prices[i]
                else:
                    zig[i] = zig[i - 1]

            # 对于最后一根K线，直接沿用倒数第2根的ZIG值（不做新的转折点判定）
            if len(prices) >= 2:
                zig[-1] = zig[-2]

            return zig

        # 计算买线（ZIG(3,10)确认版）和卖线（买线3日均线）
        买线 = calculate_zig(C, threshold_pct=10)
        卖线 = pd.Series(买线).rolling(window=3, min_periods=1).mean().values

        # 计算短周期ZIG(3,5)确认版用于波段卖信号
        zig_5 = calculate_zig(C, threshold_pct=5)

        # 收集所有信号及其位置
        buy_signals = []
        sell_signals = []
        signal_positions = []  # 每个元素为 (index, type)

        # ========== 波段买点：买线上穿卖线（只对已确认K线产生信号） ==========
        if len(买线) >= 3 and len(卖线) >= 3:
            # 只遍历到倒数第2根K线，最后一根K线不产生新信号
            for i in range(1, len(买线) - 1):
                if 买线[i] > 卖线[i] and 买线[i - 1] <= 卖线[i - 1]:
                    buy_signals.append('波段买')
                    signal_positions.append((i, '波段买'))

        # ========== 波段卖点：ZIG(3,5) 连续下跌形态（确认版） ==========
        if len(zig_5) >= 5:
            # 只遍历到倒数第2根K线
            for i in range(3, len(zig_5) - 1):
                z1 = zig_5[i]
                z2 = zig_5[i - 1]
                z3 = zig_5[i - 2]
                z4 = zig_5[i - 3]
                condition = (z1 < z2) and (z2 >= z3) and (z3 >= z4)
                if condition:
                    sell_signals.append('波段卖')
                    signal_positions.append((i, '波段卖'))

        # 确定最近一次信号
        last_signal_type = None
        last_signal_offset = None
        if signal_positions:
            # 按索引排序，取最后一个（最大索引）
            signal_positions.sort(key=lambda x: x[0])
            last_idx, last_type = signal_positions[-1]
            total_len = len(C)
            last_signal_type = last_type
            last_signal_offset = total_len - 1 - last_idx  # 距离最后一根K线的偏移

        # 计算趋势颜色
        trend_color = 'green'  # 默认下降
        if len(买线) >= 2 and len(卖线) >= 2:
            if 买线[-1] > 卖线[-1]:
                trend_color = 'red'

        # ---------- 以下为辅助指标计算（不影响主信号，仅供扩展）----------
        # 计算MA均线
        MA5 = pd.Series(C).rolling(window=5, min_periods=1).mean().values
        MA10 = pd.Series(C).rolling(window=10, min_periods=1).mean().values
        MA20 = pd.Series(C).rolling(window=20, min_periods=1).mean().values

        # 计算基线（REF(LLV(C,30),1)的2日均线）
        min_30 = pd.Series(C).rolling(window=30, min_periods=1).min().shift(1)
        基线 = min_30.rolling(window=2, min_periods=1).mean().values

        # 精准买点 (EMA交叉) - 同花顺原指标中的 "精准买" 信号，暂不加入主信号
        def ema(data, period):
            return pd.Series(data).ewm(span=period, adjust=False).mean().values

        X1 = (C + L + H) / 3
        X2 = ema(X1, 6)
        X3 = ema(X2, 5)

        # 游资进入信号
        if len(C) >= 75:
            def sma(data, period, weight=1):
                return pd.Series(data).ewm(alpha=weight / period, adjust=False).mean().values

            llv_75 = pd.Series(L).rolling(window=75, min_periods=1).min().values
            hhv_75 = pd.Series(H).rolling(window=75, min_periods=1).max().values

            denominator = hhv_75 - llv_75
            denominator = np.where(denominator == 0, 1, denominator)

            close_norm = (C - llv_75) / denominator * 100
            open_norm = (O - llv_75) / denominator * 100

            close_norm = np.where(np.isnan(close_norm), 0.0, close_norm)
            close_norm = np.where(np.isinf(close_norm) & (close_norm > 0), 100.0, close_norm)
            close_norm = np.where(np.isinf(close_norm) & (close_norm < 0), 0.0, close_norm)

            open_norm = np.where(np.isnan(open_norm), 0.0, open_norm)
            open_norm = np.where(np.isinf(open_norm) & (open_norm > 0), 100.0, open_norm)
            open_norm = np.where(np.isinf(open_norm) & (open_norm < 0), 0.0, open_norm)

            sma1_close = sma(close_norm, 20, 1)
            sma2_close = sma(sma1_close, 15, 1)
            VARF1 = 100 - 3 * sma1_close + 2 * sma2_close

            sma1_open = sma(open_norm, 20, 1)
            sma2_open = sma(sma1_open, 15, 1)
            VAR101 = 100 - 3 * sma1_open + 2 * sma2_open

            if len(VARF1) >= 2 and len(VAR101) >= 2 and len(V) >= 2:
                VAR111 = (VARF1[-1] < VAR101[-2] and V[-1] > V[-2] and C[-1] > C[-2])

                count_signals = 0
                for i in range(max(0, len(VARF1) - 30), len(VARF1)):
                    if i >= 1 and i < len(V) - 1:
                        if (VARF1[i] < VAR101[i - 1] and V[i] > V[i - 1] and C[i] > C[i - 1]):
                            count_signals += 1

        # 超买超卖信号（RSI）
        def calculate_rsi(prices, period=14):
            if len(prices) < period + 1:
                return np.zeros(len(prices))

            deltas = np.diff(prices)
            seed = deltas[:period]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 0
            rsi = np.zeros_like(prices)
            rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

            for i in range(period, len(prices)):
                delta = deltas[i - 1]
                if delta > 0:
                    upval = delta
                    downval = 0.0
                else:
                    upval = 0.0
                    downval = -delta

                up = (up * (period - 1) + upval) / period
                down = (down * (period - 1) + downval) / period
                rs = up / down if down != 0 else 0
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)

            return rsi

        # 去重
        buy_signals = list(set(buy_signals))
        sell_signals = list(set(sell_signals))

        if log_debug and (buy_signals or sell_signals):
#             log.info(f"THS指标信号 {stock}: 买={buy_signals}, 卖={sell_signals}, 趋势={trend_color}")

        # 返回结果
            pass
        result = {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'trend_color': trend_color,
            'last_signal_type': last_signal_type,
            'last_signal_offset': last_signal_offset
        }
        return result

    except Exception as e:
#         log.error(f"计算同花顺指标时出错 {stock}: {str(e)}")
        import traceback
#         log.error(traceback.format_exc())
        return {
            'buy_signals': [],
            'sell_signals': [],
            'trend_color': 'green',
            'last_signal_type': None,
            'last_signal_offset': None
        }


def check_volume_drop_signal(stock, context):
    """
    检测放量大跌信号（跌幅6%以上且放量）
    判定条件：当前相对昨日收盘跌幅≥6% + 估算全天成交量≥5日均量1.5倍
    """
    try:
        # 1. 获取实时行情数据，快速过滤无效标的
        current_data = get_current_data()
        if stock not in current_data:
            return False
        stock_quote = current_data[stock]
        current_price = stock_quote.last_price
        today_open = stock_quote.day_open
        # 过滤价格为0的异常行情
        if current_price == 0 or today_open == 0:
            return False

        # 2. 获取近2日K线数据，提取昨日收盘价（需至少2条有效数据）
        hist_2d = attribute_history(stock, 2, '1d',
                                    ['open', 'close', 'volume'],
                                    skip_paused=True)
        if len(hist_2d) < 2:
            return False
        yesterday_close = hist_2d['close'].iloc[-2]  # 用iloc更安全，适配pandas
        if yesterday_close == 0:
            return False

        # 3. 计算当前跌幅，未达6%直接返回（提前过滤，提升效率）
        drop_pct = (yesterday_close - current_price) / yesterday_close
        if drop_pct < 0.06:
            return False

        # 4. 获取近6日成交量数据，计算5日均量（排除今日）
        vol_6d = attribute_history(stock, 6, '1d', ['volume'], skip_paused=True)
        # 修正：取6日数据需至少6条，否则无法提取前5日有效数据
        if len(vol_6d) < 6:
            return False
        # 前5日成交量（排除今日最后1条），计算均值
        vol_5d = vol_6d['volume'].iloc[:-1]
        avg_volume_5 = vol_5d.mean()
        # 补全除零保护：5日均量为0时直接返回（避免后续除法报错）
        if avg_volume_5 < 1:
            return False
        # 今日已成交成交量
        today_volume = vol_6d['volume'].iloc[-1]
        if today_volume < 1:
            return False

        # 5. 计算当前已交易分钟数（适配A股交易时间：9:30-11:30、13:00-15:00）
        current_time = context.current_dt.time()
        market_minutes = 0
        # 上午交易时段
        if dt.time(9, 30) <= current_time <= dt.time(11, 30):
            market_minutes = (current_time.hour - 9) * 60 + current_time.minute - 30
        # 下午交易时段
        elif dt.time(13, 0) <= current_time <= dt.time(15, 0):
            market_minutes = 120 + (current_time.hour - 13) * 60 + current_time.minute
        # 午休/收盘后，直接取对应时段的总分钟数
        elif dt.time(11, 30) < current_time < dt.time(13, 0):
            market_minutes = 120
        elif current_time > dt.time(15, 0):
            market_minutes = 240
        # 非交易时间（开盘前），直接返回
        if market_minutes < 1:
            return False

        # 6. 估算全天成交量（加早盘修正因子，适配早盘量能偏高特性）
        estimated_daily_volume = today_volume * (240 / market_minutes)
        # 修正因子：早盘前60分钟1.2→0.8，上午后60分钟0.8→1.0，下午保持1.0
        if market_minutes < 60:
            estimated_daily_volume *= 1.2 - (market_minutes / 60) * 0.4
        elif 60 <= market_minutes < 120:
            estimated_daily_volume *= 0.8 + ((market_minutes - 60) / 60) * 0.2

        # 7. 计算量比，判定是否放量
        volume_ratio = estimated_daily_volume / avg_volume_5
        if volume_ratio >= 1.5:
#             log.info(f"🚨 放量大跌信号: {stock} | 跌幅: {drop_pct * 100:.2f}% | "
#                      f"量比: {volume_ratio:.2f}倍 | 已交易: {market_minutes}分钟")
            return True

        return False

    except Exception as e:
        # 修复：原代码少闭合括号的语法错误，优化日志提示
#         log.error(f"检测{stock}放量大跌信号出错: {str(e)}")
        return False


def sell_limit_per5min(context):
    """
    每5分钟检测持仓股票是否需要卖出
    基于同花顺指标的波段卖信号和放量大跌信号
    增加涨停不卖判断
    """
    # 初始化交易记录
    if not hasattr(g, 'today_trades'):
        g.today_trades = []

    current_data = get_current_data()
    date = transform_date(context.previous_date, 'str')

    for stock in list(context.portfolio.positions):
        # 排除货币基金等
        if stock == '511880.XSHG':
            continue

        position = context.portfolio.positions[stock]
        if position.closeable_amount == 0:
            continue

        # 检查是否停牌
        if current_data[stock].paused:
            continue

        try:
            # 获取当前价格和成本价（确保为数值类型）
            current_price = current_data[stock].last_price
            avg_cost = position.avg_cost
            high_limit = current_data[stock].high_limit

            # 数值有效性检查
            if current_price is None or avg_cost is None or high_limit is None:
#                 log.warning(f"{stock} 价格数据缺失，跳过")
                continue
            if not (isinstance(current_price, (int, float)) and current_price > 0):
#                 log.warning(f"{stock} 当前价格无效: {current_price}")
                continue
            if not (isinstance(avg_cost, (int, float)) and avg_cost > 0):
#                 log.warning(f"{stock} 成本价无效: {avg_cost}")
                continue

            # ==================== 新增：涨停不卖判断 ====================
            if high_limit > 0 and current_price >= high_limit * 0.995:
                # log.info(f"{stock} 当前涨停，暂不卖出")
                continue

            # 1. 检测同花顺指标信号（可能返回复杂结构，需异常保护）
            try:
                ths_signals = calculate_ths_indicators(stock, context, 30, '5m')
                # 优化：仅在下降趋势且最近3根K线内有波段卖信号时，才视为有效卖出信号
                if (ths_signals['trend_color'] == 'green' and
                        '波段卖' in ths_signals.get('sell_signals', []) and
                        ths_signals.get('last_signal_offset', 100) <= 2):
                    pass
                    has_sell_signal = True
                else:
                    has_sell_signal = False
            except Exception as e:
#                 log.error(f"{stock} 计算同花顺指标异常: {e}")
                has_sell_signal = False

            # 2. 检测放量大跌信号
            try:
                has_volume_drop = check_volume_drop_signal(stock, context)
            except Exception as e:
#                 log.error(f"{stock} 检测放量大跌异常: {e}")
                has_volume_drop = False

            # 3. 紧急止损：波段卖信号 + 放量大跌
            if has_sell_signal and has_volume_drop:
                loss_pct = (avg_cost - current_price) / avg_cost
                details = {
                    '触发信号': '波段卖 + 放量大跌',
                    '成本价': f"{avg_cost:.2f}",
                    '当前价': f"{current_price:.2f}",
                    '亏损比例': f"{loss_pct:.2%}",
                    '同花顺信号': ths_signals.get('sell_signals', [])
                }
                # 记录交易
                record_sell_trade(context, stock, "紧急止损-波段卖+放量大跌", details, current_data, date)
                # 执行卖出（聚宽标准：order_target_value(security, value)）
                order_target_value(stock, 0)
#                 log.info(f"★★★ 紧急止损 ★★★ 股票: {stock}, 亏损: {loss_pct:.2%}")

            # 4. 波段卖出：仅波段卖信号且亏损超过5%，且量比大于g.max_sell_vol_ratio
            elif has_sell_signal:
                loss_pct = (avg_cost - current_price) / avg_cost
                if loss_pct >= 0.05:
                    vol_ratio = get_5min_volume_ratio(stock, context, period=5)
                    if vol_ratio is not None and vol_ratio > g.max_sell_vol_ratio:
                        details = {
                            '触发信号': '波段卖 + 亏损5%+（放量确认）',
                            '成本价': f"{avg_cost:.2f}",
                            '当前价': f"{current_price:.2f}",
                            '亏损比例': f"{loss_pct:.2%}",
                            '同花顺信号': ths_signals.get('sell_signals', []),
                            '5分钟量比': f"{vol_ratio:.2f}"
                        }
                        record_sell_trade(context, stock, "波段卖出", details, current_data, date)
                        order_target_value(stock, 0)
#                         log.info(f"波段卖出（放量）: {stock}, 亏损: {loss_pct:.2%}, 量比: {vol_ratio:.2f}")
                    else:
#                         log.info(f"{stock} 波段卖信号但未放量（量比{vol_ratio}），暂不卖出")

            # 5. 放量大跌：跌幅达到8%以上
                        pass
            elif has_volume_drop:
                # 正确获取昨日收盘价（使用前复权数据）
                hist = attribute_history(stock, 2, '1d', ['close'], skip_paused=True, df=True)
                if hist is None or len(hist) < 2:
#                     log.warning(f"{stock} 历史数据不足，无法计算跌幅")
                    continue
                yesterday_close = hist['close'].iloc[-2]  # 前一交易日的收盘价
                drop_pct = (yesterday_close - current_price) / yesterday_close
                if drop_pct >= 0.08:
                    details = {
                        '触发信号': '放量大跌8%+',
                        '昨收价': f"{yesterday_close:.2f}",
                        '当前价': f"{current_price:.2f}",
                        '跌幅': f"{drop_pct:.2%}"
                    }
                    record_sell_trade(context, stock, "放量大跌止损", details, current_data, date)
                    order_target_value(stock, 0)
#                     log.info(f"放量大跌止损: {stock}, 跌幅: {drop_pct:.2%}")

        except ValueError as ve:
            # 捕获数值转换错误，很可能就是股票代码被误转换
#             log.error(f"处理股票 {stock} 卖出检测时发生数值转换错误: {ve}")
            continue
        except Exception as e:
#             log.error(f"处理股票 {stock} 卖出检测时发生未知错误: {str(e)}")
            continue


def get_hot_leader_first_yin_stocks(context, date_1, date):
    """获取热门龙头首阴股票（策略2中的关键函数）"""
    try:
        # log.info("=" * 70)
        # log.info("? 开始筛选热门龙头首阴股票")
        # log.info(f"   前2交易日: {date_1} (查找连板龙头)")
        # log.info(f"   前1交易日: {date} (判断断板)")
        # log.info("=" * 60)

        # # 从热门概念缓存提取龙头股票
        # log.info("? 从热门概念缓存提取龙头股票")
        # prev_date_str = date_1.strftime('%Y%m%d')
        # log.info(f"   查询日期: {prev_date_str}")

        # 获取该日期的热门概念
        hot_concepts = []
        if hasattr(g, 'hot_concepts_data_cache') and prev_date_str in g.hot_concepts_data_cache:
            hot_concepts = g.hot_concepts_data_cache.get(prev_date_str, [])

        # log.info(f"   缓存概念数: {len(hot_concepts)}")

        # 提取各概念的龙头股票
        leader_stocks = []
        for concept in hot_concepts:
            if not concept or not isinstance(concept, dict):
                continue

            concept_name = concept.get('name', '')
            stock_list = concept.get('stock_list', [])

            # 筛选连板龙头
            for stock_item in stock_list:
                if not stock_item:
                    continue

                stock_code = stock_item.get('code', '')
                if not stock_code:
                    continue

                # 检查连板数
                try:
                    ccd = get_continue_count_df([stock_code], date_1, 5)
                    if len(ccd) > 0 and ccd['count'].iloc[0] >= 2:
                        leader_stocks.append(stock_code)
                        stock_name = get_security_info(stock_code).display_name
                        # log.info(f"   概念 [{concept_name}] 龙头 (连板{ccd['count'].iloc[0]}):")
                #         log.info(
                #             f"     ? {stock_name} ({stock_code.split('.')[0]} → {stock_code}) - {ccd['count'].iloc[0]}天{ccd['count'].iloc[0]}板")
                except:
                    continue

        # 去重
        leader_stocks = list(set(leader_stocks))
        # log.info("   --------------------------------------------------------")
        # log.info(f"   ? 统计:")
        # log.info(f"     o 有龙头的概念数: {len(hot_concepts)}/{len(hot_concepts)}")
        # log.info(f"     o 提取龙头股票数: {len(leader_stocks)}")
        # log.info("   --------------------------------------------------------")

        # 输出最终龙头股票池
        # log.info(f"   ? 最终龙头股票池:")
        for i, stock in enumerate(leader_stocks, 1):
            try:
                stock_name = get_security_info(stock).display_name
#                 log.info(f"      {i}. {stock_name} ({stock})")
            except:
#                 log.info(f"      {i}. {stock}")

#         log.info(f"   ? 找到 {len(leader_stocks)} 只热门龙头股")
#         log.info("   ------------------------------------------------------------------")

        # 筛选条件1: 前2交易日连板3板以上的换手板
        # log.info(f"   ? 筛选条件1: 前2交易日连板3板以上的换手板，5元以下可以有一字板")
                pass
        qualified_stocks = []

        for stock in leader_stocks:
            try:
                stock_name = get_security_info(stock).display_name
                ccd = get_continue_count_df([stock], date_1, 5)

                if len(ccd) == 0:
                    # log.debug(f"     ? {stock_name} ({stock}) - 无连板数据")
                    continue

                # 检查连板数
                board_count = ccd['count'].iloc[0]
                extreme_count = ccd['extreme_count'].iloc[0]

                if board_count < 3:
                    # log.debug(f"     ? {stock_name} ({stock}) - 连板{board_count}板 (不足3板)")
                    continue

                # 检查是否为一字板
                price_data = attribute_history(stock, board_count, '1d',
                                               ['open', 'close', 'high', 'low'],
                                               skip_paused=True)

                has_non_extreme = False
                for i in range(len(price_data)):
                    if price_data['open'][i] < price_data['high'][i]:
                        has_non_extreme = True
                        break

                # 检查股价
                current_price = get_price([stock], end_date=date_1, fields=['close'], count=1)['close'][0]

                if has_non_extreme or current_price < 5:
                    qualified_stocks.append(stock)
                    # log.debug(f"     ✓ {stock_name} ({stock}) - 符合条件")
                else:
                    # log.debug(f"     ? {stock_name} ({stock}) - 前2日为一字板(非换手板)")

                    pass
            except Exception as e:
                # log.debug(f"     ? {stock} - 筛选失败: {str(e)}")
                continue

        if not qualified_stocks:
            # log.warning(f"   ? 未找到符合连板条件的股票")

            pass
        return qualified_stocks

    except Exception as e:
#         log.error(f"筛选热门龙头首阴股票失败: {str(e)}")
        import traceback
#         log.error(traceback.format_exc())
        return []


# 2. 盘前数据统计函数
def record_morning_stats(context):
    """
    盘前环境评估。

    作用很像“今天开盘前的作战会议”：
    - 先重置昨天留下来的状态；
    - 再看大盘最近的涨跌、波动、量能；
    - 最后给今天定一个大方向，比如 strong_up / up / flat / down。

    后面很多优先级和过滤条件，都会参考这里写进 g.dynamic_params 的结果。
    """
    try:
#         log.info(f"====== {context.current_dt.strftime('%Y-%m-%d')} 盘前数据 ======")
        # 每日重置选股完成标志，确保每天重新选股
        g.stock_list_done = False
        g.is_empty = False  # 新增：每日重置空仓标志

        # ========== 修复点2：每日重置涨停标记 ==========
        if not hasattr(g, 'stocks_limit_up_today'):
            g.stocks_limit_up_today = set()
        else:
            g.stocks_limit_up_today.clear()

        # 清空个股积分缓存
        clear_score_cache(context)

        # 获取大盘数据
        index_data = attribute_history('000001.XSHG', 5, '1d',
                                       ['close', 'volume'], skip_paused=True)

        if not index_data.empty and len(index_data) >= 2:
            current_close = index_data['close'].iloc[-1]
            prev_close = index_data['close'].iloc[-2]
            change_rate = (current_close - prev_close) / prev_close * 100

            # 计算波动率
            volatility = index_data['close'].pct_change().std() * 100

            # 计算量能比
            current_volume = index_data['volume'].iloc[-1]
            avg_volume = index_data['volume'].head(-1).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # 判断市场趋势
            if change_rate > 1:
                trend = "strong_up"
            elif change_rate > 0:
                trend = "up"
            elif change_rate > -1:
                trend = "flat"
            else:
                trend = "down"

            log.info("=========市场状况=========")
            log.info(f"- 大盘趋势: {trend}")
            # log.info(f"- 波动率: {volatility:.2f}%")
            # log.info(f"- 量能比: {volume_ratio:.2f}")

            g.dynamic_params = {
                'trend': trend,
            }

            # 存储市场统计
            g.trade_stats['market_stats'] = {
                'date': context.current_dt.strftime('%Y-%m-%d'),
                'trend': trend,
                'change_rate': change_rate,
                'volatility': volatility,
                'volume_ratio': volume_ratio
            }
            update_strategy_priority(trend)

    except Exception as e:
#         log.error(f"记录盘前统计失败: {str(e)}")


        pass
def get_last_n_auction_avg(s, end_date, n=5):
    # 获取最近n个交易日（含end_date）
    trade_days = get_trade_days(end_date=end_date, count=n)
    # 获取这n日的集合竞价量
    auction_data = get_call_auction([s], start_date=trade_days[0], end_date=trade_days[-1], fields=['time', 'volume'])
    if auction_data.empty:
        return 0
    return auction_data['volume'].mean()


def sell_limit_down(context):
    """
    开盘前的持仓预警卖出函数。

    它主要处理一种情况：
    昨天K线已经很难看，比如放量长上影，说明上方卖压重；
    如果今天集合竞价和开盘又不够强，就优先考虑先卖。

    简单理解：
    这是“开盘前先排雷”，不是全天主卖出逻辑。
    """
    date = context.previous_date
    current_data = get_current_data()
    slist = list(context.portfolio.positions)

    # 获取集合竞价数据
    date_now = context.current_dt.strftime("%Y-%m-%d")
    auction_start = date_now + ' 09:15:00'
    auction_end = date_now + ' 09:25:00'
    auctions = get_call_auction(
        slist, start_date=auction_start, end_date=auction_end,
        fields=['time', 'current', 'volume']
    ).set_index('code') if slist else None

    for stock in slist:
        if stock == '511880.XSHG':  # 排除基金
            continue
        position = context.portfolio.positions[stock]

        try:
            # 获取K线数据
            price_data = get_price(stock, end_date=date, count=6,
                                   fields=['open', 'close', 'high', 'low', 'volume', 'high_limit'], skip_paused=False)
            if price_data.shape[0] < 6:
                continue
            # 放量长上影昨日数据
            yest = price_data.iloc[-1]
            # 涨停放量低开昨日数据
            prev = price_data.iloc[-1]
            # print(f"price_data:{price_data}, yest:{yest}, prev{prev}")
            avg_vol_5 = price_data['volume'].iloc[:-1].mean()

            prev_close = prev['close']
            prev_high_limit = prev['high_limit']
            prev_volume = prev['volume']
            today_open = current_data[stock].day_open
            if pd.isna(today_open) or today_open == 0:
                continue

            # 估值数据
            valuation = get_valuation(stock, date, date, fields=['pe_ratio', 'circulating_market_cap'])
            if valuation.empty:
                continue
            pe_ratio = valuation['pe_ratio'].iloc[0]
            market_cap = valuation['circulating_market_cap'].iloc[0]

            # 集合竞价量
            auction_volume = auctions.loc[stock]['volume'] if (auctions is not None and stock in auctions.index) else 1
            avg_auction_vol_5 = get_last_n_auction_avg(stock, date_now, n=5) or 1  # 防止为0
            auction_vol_ratio = auction_volume / avg_auction_vol_5 if avg_auction_vol_5 > 0 else 1

            # === 优先级2：放量长上影开盘卖出 (优化版) ===
            upper_shadow = yest['high'] - max(yest['open'], yest['close'])
            lower_shadow = min(yest['open'], yest['close']) - yest['low']
            real_body = abs(yest['close'] - yest['open'])
            total_range = yest['high'] - yest['low']

            # 基础的“放量长上影”定义
            is_long_upper_shadow = (
                    upper_shadow > lower_shadow * 1.2 and
                    upper_shadow > 1.5 * real_body and
                    upper_shadow > 0.3 * total_range
            )
            is_big_vol = yest['volume'] > 1.5 * avg_vol_5

            yesterday_close = yest['close']
            # 当today_open为0时，直接将open_change赋值为-100，避免后续计算报错
            if today_open == 0:
                open_change = -100
            else:
                open_change = (today_open / yesterday_close - 1) * 100
            auction_vol_ratio = auction_volume / avg_auction_vol_5 if avg_auction_vol_5 > 0 else 1
            # 首先，记录所有符合基础定义的信号
            if is_long_upper_shadow and is_big_vol:
                # --- 优化后的卖出条件 ---
                # 条件1: 上影线远大于下影线，确认T-1日卖压沉重
                is_seller_dominant = upper_shadow > lower_shadow * 3

                # 条件2: 开盘涨幅不能过高，避免卖在强势反转的起点
                is_open_not_strong = open_change < 2

                # 条件3 (核心优化): 动态判断竞价量
                # 逻辑：如果开盘价疲软（平开或微涨），我们可以容忍稍大的竞价量，
                # 因为“放量不涨”本身就是滞涨或出货的信号。
                is_auction_ok = False
                if open_change <= 0.5:  # 如果开盘价非常弱势
                    # 容忍2倍以下的竞价放量，视为多空分歧加大但多头无力上攻
                    if auction_vol_ratio < 2.0:
                        is_auction_ok = True
                else:  # 如果开盘价稍强 (0.5% < open_change < 2%)
                    # 此时要求更严格的竞价量，必须是缩量或微量，防止是强势承接
                    if auction_vol_ratio < 1.4:
                        is_auction_ok = True

                if (
                        position.closeable_amount > 0 and
                        is_seller_dominant and
                        is_open_not_strong and
                        is_auction_ok  # 使用新的动态条件
                ):
                    order_target_value(stock, 0)
                    date_str = date.strftime('%Y-%m-%d')
                    stock_name = get_security_info(stock, date_str).display_name
#                     log.info(
#                         f"卖出结果：{stock}({stock_name}) 触发放量长上影卖出，"
#                         f"上影/下影{upper_shadow / lower_shadow if lower_shadow > 0 else 'inf':.2f}，"
#                         f"竞价量比{auction_vol_ratio:.2f}"
#                     )
                continue
        except Exception as e:
#             log.error(f"处理股票 {stock} 时出错: {e}")
            continue


# 判断是否需要空仓
def should_empty_position(context):
    """
    判断今天是否直接进入“空仓模式”。

    设计目的：
    市场环境太差时，今天就不选股、不买入，优先保护资金。

    当前实际启用的条件：
    1. 沪深300量能异常（大于前4日均量2倍或小于0.5倍）。

    注意：
    注释里提到的其它空仓想法不一定都已实现，真正生效的条件以函数主体为准。
    """
    # ========== 1. 量能异常检测 ==========
    try:
        volume_data = attribute_history('000300.XSHG', 5, '1d', fields=['volume'], skip_paused=True)
        if len(volume_data) >= 5:
            avg_volume = volume_data['volume'][:-1].mean()
            current_volume = volume_data['volume'][-1]
            if current_volume > 2 * avg_volume or current_volume < 0.5 * avg_volume:
#                 log.info('空仓信号：大盘量能异常（>2倍或<0.5倍）')
                return True
    except Exception as e:
#         log.error(f"获取大盘量能数据出错: {e}")

        pass
    return False


# 选股
def format_stock_list_with_names(stock_list, current_data=None):
    """Format stock codes as code(name) for logs/messages."""
    if not stock_list:
        return ''

    if current_data is None:
        try:
            current_data = get_current_data()
        except Exception:
            current_data = {}

    formatted = []
    for stock in stock_list:
        stock_name = stock
        try:
            if stock in current_data and hasattr(current_data[stock], 'name') and current_data[stock].name:
                stock_name = current_data[stock].name
            else:
                stock_info = get_security_info(stock)
                if stock_info and getattr(stock_info, 'display_name', None):
                    stock_name = stock_info.display_name
        except Exception:
            pass
        formatted.append(f"{stock}({stock_name})")
    return ','.join(formatted)


def get_stock_list(context):
    """
    每日选股主入口。

    可以把它理解为“先粗筛，再分组”的步骤：
    1. 判断今天是否空仓；
    2. 准备基础股票池；
    3. 围绕昨日涨停股，拆出多个策略模式；
    4. 把结果写入 g，供后面买入逻辑使用。
    """
#     log.info(f"[{context.current_dt.strftime('%H:%M:%S')}] get_stock_list 开始执行")
    # 如果已经完成选股，直接返回
    if getattr(g, 'stock_list_done', False):
#         log.info("今日已完成选股，跳过重复执行")
        return

    # 判断是否需要空仓（原有逻辑）
    if should_empty_position(context):
        current_data = get_current_data()
        for stock in context.portfolio.positions:
            log.info(f"[空仓] 卖出持仓股: {stock}")
#             log.info(f"[空仓] 卖出持仓股: {stock}")
            send_message(f"卖出持仓股: {stock}")
            order_target_value(stock, 0)
        g.is_empty = True
        # 新增：清空所有候选股票列表，避免旧数据影响
        g.qualified_stocks = []
        g.lblt_stocks = []
        g.rzq_stocks = []
        g.gk_stocks = []
        g.dk_stocks = []
        g.fxsbdk_stocks = []
        g.score_cache = {}
        log.info("[空仓] 当前满足空仓条件，今日不参与选股，已清空候选列表")
#         log.info("[空仓] 当前满足空仓条件，今日不参与选股，已清空候选列表。")
        g.stock_list_done = True
        return
    else:
        g.is_empty = False

    # previous_date 是上一个交易日，不一定等于自然日的“昨天”。
    date_now = context.current_dt.strftime("%Y-%m-%d")
    date = context.previous_date
    date_2, date_1, date = get_trade_days(end_date=date, count=3)

    # 初始列表 = 全市场股票，经过新股 / ST / 科创北交等基础过滤后的结果
    initial_list = prepare_stock_list(date)

    # 本策略围绕“涨停股”展开，所以昨日涨停列表是后续分组的起点
    hl0_list = get_hl_stock(initial_list, date)

    # 如果昨日涨停列表为空，则直接设置空结果并完成
    if not hl0_list:
        g.gap_up = []
        g.gap_down = []
        g.reversal = []
        g.fxsbdk = []
        g.lblt = []
        g.stock_list_done = True
#         log.info("昨日涨停股票为空，选股完成")
        return

    # 前日曾涨停
    hl1_list = get_ever_hl_stock(initial_list, date_1)
    # 前前日曾涨停
    hl2_list = get_ever_hl_stock(initial_list, date_2)

    elements_to_remove = set(hl1_list + hl2_list)
    get_all_hot_concepts_optimized(context)

    # 先算出几类不依赖集合竞价的模式池
    g.gap_up = [stock for stock in hl0_list if stock not in elements_to_remove]
    g.gap_down = [s for s in hl0_list if s not in hl1_list]
    h1_list = get_ever_hl_stock2(initial_list, date)
    elements_to_remove = get_hl_stock(initial_list, date_1)
    broken_leaders = get_dblt_stocks(hl1_list, date_1, date, context)
    # log.info(f"市场最高标龙首阴候选broken_leaders：{broken_leaders}")
    hot_leader_first_yin = get_hot_leader_first_yin_stocks(context, date_1, date)
    g.reversal = list(set(h1_list) - set(elements_to_remove))
    g.reversal = list(set(g.reversal + hot_leader_first_yin))
    for stock in broken_leaders:
        if stock not in g.reversal:
            g.reversal.append(stock)
            # log.info(f"   ➕ {stock} | 新增断板龙头")
    g.fxsbdk = get_ll_stock(initial_list, date)

    # 集合竞价常用于判断开盘强弱，是打板策略里的关键参考
    auction_start = date_now + ' 09:15:00'
    auction_end = date_now + ' 09:25:00'
    auctions = get_call_auction(hl0_list, start_date=auction_start, end_date=auction_end,
                                fields=['time', 'current']).set_index('code')

    # 关键修改：如果集合竞价数据为空且昨日涨停列表非空，则本次不完成选股（等待重试）
    if auctions.empty:
#         log.info("集合竞价数据为空，稍后重试")
        return  # 不设置完成标志，等待下一次定时任务

    # 获取前收盘价
    h = get_price(hl0_list, end_date=date, fields=['close'], count=1, panel=False).set_index('code')
    if h.empty:
        g.lblt = []
        g.stock_list_done = True
        return

    auctions['pre_close'] = h['close']
    gk_list = auctions.query('pre_close * 1.00 < current').index.tolist()
    gkb = len(gk_list) / len(hl0_list) * 100

    if gkb < 76:
        g.lblt = []
    else:
        g.lblt = hl0_list
        # 高风险一字板筛选
        high_risk_stocks = []
        lblt_stocks = []
        for stock in g.lblt:
            try:
                if has_consecutive_extreme_limit(stock, date):
                    high_risk_stocks.append(stock)
                else:
                    lblt_stocks.append(stock)
            except Exception as e:
                # log.error(f"计算{stock}近期一字板失败: {str(e)}")
                pass
        if high_risk_stocks:
            # log.info(
            #     f"近5日有2个以上连续一字板的高风险个股(将被排除): {[get_security_info(s).display_name for s in high_risk_stocks]}")
            pass
        g.lblt = lblt_stocks
        g.high_risk_stocks_today = high_risk_stocks  # 缓存高风险股票，供 buy() 复用

    # 到这里说明“今日候选池”已经准备完成，后续买入阶段可直接使用
    g.stock_list_done = True
#     log.info("选股完成")
    buy(context)


def get_dblt_stocks(hl1_list, date_1, date, context):
    """
    识别“断板龙头”。

    什么叫断板：
    一只连续涨停的强势股，到了下一天没有继续封住涨停，就叫断板。

    这个函数做的事是：
    1. 找出前一阶段的最高连板股；
    2. 检查它们在下一天是否断板；
    3. 再根据断板方式分类，判断有没有后续做弱转强的价值。
    """
    ccd = get_continue_count_df(hl1_list, date_1, 20) if len(hl1_list) != 0 else pd.DataFrame(
        index=[],
        data={'count': [], 'extreme_count': []}
    )

    # 最高连板数
    M = ccd['count'].max() if len(ccd) != 0 else 0

    # log.info(f"   前第2日({date_1})连板股票数：{len(ccd)} 只")
    # log.info(f"   最高连板数：{M} 板")

    if M == 0 or len(ccd) == 0:
#         log.info("⚠️ 无连板数据")
        g.target_list = []
        return []

    # ==================== 第五步：识别断板龙头 ====================
    # log.info("🔍 步骤3：识别断板龙头...")

    # 获取最高板股票列表
    max_board_df = ccd[ccd['count'] == M]
    max_board_stocks = list(max_board_df.index)

    # log.info(f"   最高板股票数：{len(max_board_stocks)} 只")

    # 获取前第1日的涨停股票列表
    initial_list_date = prepare_stock_list(date)
    hl_list_date = get_hl_stock(initial_list_date, date)

    # log.info(f"   前第1日({date})涨停股：{len(hl_list_date)} 只")

    # 识别断板龙头
    broken_leaders = []
    broken_leaders_info = {}

    for stock in max_board_stocks:
        # 检查该股票在前第1日是否涨停
        if stock not in hl_list_date:
            # 断板
            broken_leaders.append(stock)

            # 获取详细信息
            try:
                stock_info = get_security_info(stock, date)
                stock_name = stock_info.display_name if stock_info else stock

                # 获取行情数据
                perf_data = get_price(
                    stock,
                    start_date=date,
                    end_date=date,
                    fields=['open', 'close', 'high', 'low', 'high_limit'],
                    skip_paused=False,
                    fq='pre',
                    panel=False
                )

                if not perf_data.empty:
                    open_price = perf_data['open'].iloc[0]
                    close_price = perf_data['close'].iloc[0]
                    high_price = perf_data['high'].iloc[0]
                    low_price = perf_data['low'].iloc[0]
                    high_limit = perf_data['high_limit'].iloc[0]

                    # 获取前第2日收盘价
                    prev_data = get_price(
                        stock,
                        start_date=date_1,
                        end_date=date_1,
                        fields=['close'],
                        skip_paused=True,
                        fq='pre',
                        panel=False
                    )

                    if not prev_data.empty:
                        prev_close = prev_data['close'].iloc[0]
                        change_pct = (close_price - prev_close) / prev_close * 100
                    else:
                        change_pct = 0

                    # 判断断板类型
                    broken_type = classify_broken_type(
                        open_price, close_price, high_price,
                        low_price, high_limit, change_pct
                    )

                    # 保存信息
                    broken_leaders_info[stock] = {
                        'name': stock_name,
                        'board_count': int(max_board_df.loc[stock]['count']),
                        'extreme_count': int(max_board_df.loc[stock]['extreme_count']),
                        'change_pct': change_pct,
                        'broken_type': broken_type,
                        'close': close_price
                    }

                    # log.info(f"   ✅ {stock_name} ({stock}) | {broken_type} | {change_pct:+.2f}%")

            except Exception as e:
#                 log.debug(f"   ⚠️ {stock} 获取详情失败: {e}")

#     log.info(f"\n   断板龙头数：{len(broken_leaders)} 只")

    # 保存到全局变量
                pass
    g.broken_leaders = broken_leaders
    g.broken_leaders_info = broken_leaders_info

    # 筛选优质断板龙头
    priority_types = ["一字炸板-强势", "冲高回落-强势", "低开反抽-强势", "低开反抽-弱势"]
    g.priority_broken_leaders = [
        stock for stock, info in broken_leaders_info.items()
        if info['broken_type'] in priority_types
    ]

    # log.info(f"   优质断板龙头：{len(g.priority_broken_leaders)} 只")
    return broken_leaders


def has_consecutive_extreme_limit(stock, date):
    """检查股票是否有2个以上连续一字板"""
    try:
        # 获取近5个交易日的数据
        end_date = date
        start_date = get_trade_days(end_date=end_date, count=5)[0]
        price_data = get_price(stock, start_date=start_date, end_date=end_date,
                               fields=['open', 'close', 'high', 'low', 'high_limit', 'low_limit'])

        # 计算一字板天数
        consecutive_extreme = 0
        max_consecutive = 0

        for i in range(len(price_data)):
            # 判断是否为一字板
            is_extreme = (price_data['high'][i] == price_data['low'][i] and
                          price_data['close'][i] == price_data['high_limit'][i])

            if is_extreme:
                consecutive_extreme += 1
            else:
                # 更新最大连续一字板天数
                max_consecutive = max(max_consecutive, consecutive_extreme)
                consecutive_extreme = 0

        # 最后一次检查
        max_consecutive = max(max_consecutive, consecutive_extreme)

        # 如果有2个以上连续一字板，返回True
        return max_consecutive >= 2

    except Exception as e:
#         log.error(f"计算{stock}近期一字板失败: {str(e)}")
        return False


def calculate_sentiment_score_optimized(stock, context):
    """
    计算“市场情绪”因子分。

    这项分数不是看个股本身有多强，而是看它所在的外部环境是否友好。
    目前主要看三件事：
    1. 大盘最近几天是否偏强；
    2. 大盘成交量是否放大；
    3. 个股最近几天是否跑赢大盘。
    """
    try:
        # 获取市场整体情况（优先读全局缓存，避免每只股票重复调用）
        index_data = getattr(g, 'index_data_today', None)
        if index_data is None:
            index_data = attribute_history('000001.XSHG', 5, '1d',
                                           ['close', 'volume'],
                                           skip_paused=True)

        if index_data is None or index_data.empty:
            return 0

        score = 0

        # 1. 大盘趋势
        if len(index_data) >= 3:
            recent_closes = index_data['close'].tail(3)
            if recent_closes.iloc[-1] > recent_closes.iloc[-2] > recent_closes.iloc[-3]:
                score += 2  # 连续上涨
            elif recent_closes.iloc[-1] > recent_closes.iloc[-2]:
                score += 1  # 昨日上涨

        # 2. 大盘量能
        if len(index_data) >= 5:
            recent_volume = index_data['volume'].tail(2).mean()
            avg_volume = index_data['volume'].head(-2).mean()

            if recent_volume > avg_volume * 1.2:
                score += 2  # 明显放量
            elif recent_volume > avg_volume * 1.1:
                score += 1  # 适度放量

        # 3. 个股相对强度
        # 同样上涨时，能跑赢大盘的票，通常更值得关注。
        try:
            stock_data = attribute_history(stock, 5, '1d', ['close'], skip_paused=True)
            if not stock_data.empty and len(stock_data) >= 3:
                stock_change = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[-3] - 1) * 100
                index_change = (index_data['close'].iloc[-1] / index_data['close'].iloc[-3] - 1) * 100

                if stock_change > index_change + 2:  # 跑赢大盘2%以上
                    score += 1
        except:
            pass

        return min(score, 5)  # 最高5分

    except Exception as e:
#         log.error(f"计算情绪评分失败 {stock}: {str(e)}")
        return 0


def is_limit_up_open(stock, current_data):
    """检查股票是否开盘涨停"""
    try:
        if stock not in current_data:
            return False

        stock_data = current_data[stock]
        day_open = stock_data.day_open
        high_limit = stock_data.high_limit

        # 考虑微小的价格误差
        return abs(day_open - high_limit) < 0.01

    except Exception as e:
#         log.error(f"检查开盘涨停失败 {stock}: {str(e)}")
        return False

def filter_by_valuation(stock_list, context):
    """
    4月专用过滤：基于 get_valuation 的多个指标组合避坑
    条件：
        1. pe_ratio > 0 且 pe_ratio <= 200
        2. pcf_ratio > 0
        3. pb_ratio > 0
    """
    if not stock_list:
        return []
    
    # 只在4,10月生效
    current_month = context.current_dt.month
    if current_month not in [4,10]:
        # log.info(f"当前月份 {current_month}，不执行4,10月专用过滤")
        return stock_list
    
    end_date = context.previous_date
    try:
        # 一次性获取所有需要的字段
        df = get_valuation(
            stock_list,
            end_date=end_date,
            count=1,
            fields=['pe_ratio', 'pcf_ratio', 'pb_ratio']
        )
        if df.empty:
            # log.warning("未获取到估值数据，跳过过滤")
            return stock_list
        
        # 构建字典加速
        pe_map = {}
        pcf_map = {}
        pb_map = {}
        turn_map = {}
        cap_map = {}
        for _, row in df.iterrows():
            code = row['code']
            pe_map[code] = row['pe_ratio']
            pcf_map[code] = row['pcf_ratio']
            pb_map[code] = row['pb_ratio']
        
        filtered = []
        removed = []
        for stock in stock_list:
            pe = pe_map.get(stock)
            pcf = pcf_map.get(stock)
            pb = pb_map.get(stock)
            
            # 逐项检查
            fail_reason = None
            if pe is None or pd.isna(pe):
                fail_reason = "PE缺失"
            elif pe <= 0:
                fail_reason = f"PE={pe:.2f}<=0(亏损)"
            elif pe > 200:
                fail_reason = f"PE={pe:.2f}>200"
            elif pcf is None or pd.isna(pcf):
                fail_reason = "PCF缺失"
            elif pcf <= 0:
                fail_reason = f"PCF={pcf:.2f}<=0(现金流为负)"
            elif pb is None or pd.isna(pb):
                fail_reason = "PB缺失"
            elif pb <= 0:
                fail_reason = f"PB={pb:.2f}<=0(资不抵债)"
            
            else:
                filtered.append(stock)
                continue
            
            # log.info(f"{stock} 过滤原因: {fail_reason}")
            removed.append(stock)
        
#         log.info(f"4月估值过滤完成：原{len(stock_list)}只，过滤{len(removed)}只，剩余{len(filtered)}只")
        return filtered
        
    except Exception as e:
#         log.error(f"估值过滤失败: {str(e)}")
        return stock_list

# 交易
def buy(context):
    """
    买入阶段的总控函数。

    它主要负责：
    1. 确认今天是否允许买；
    2. 根据市场状态调整各模式优先级；
    3. 组装不同模式的候选股票；
    4. 最后把真正下单动作交给 execute_buy。
    """
    # 新增：如果选股未完成，跳过本次买入
    if not getattr(g, 'stock_list_done', False):
#         log.info("选股尚未完成，跳过买入")
        return

    if g.is_empty:
        return

    # g.priority_config 决定多个模式同时入选时，谁排在前面
#     log.info(f"trend:{g.trade_stats['market_stats']},g.priority_config:{g.priority_config}")

    # ========== 关键修改6：更新优先级配置为策略2的配置 ==========
    # 根据市场状态动态调整优先级（策略2逻辑）
    if hasattr(g, 'trade_stats') and 'market_stats' in g.trade_stats:
        market_stats = g.trade_stats['market_stats']
        one_to_two_success_rate = market_stats.get('one_to_two_success_rate', 0)
        market_score = market_stats.get('market_score', 0)

        # 策略2的优先级配置逻辑
        if market_score >= 80 and one_to_two_success_rate < 15:
            g.priority_config = ['lb', 'rzq', 'fxsbdk', 'dk', 'yje']
#             log.info("策略优先级动态配置: lb > rzq > fxsbdk > dk > yje")
        else:
            g.priority_config = ['lb', 'rzq', 'yje', 'fxsbdk', 'dk']

    # 这些时间变量后面会反复用到，先统一算好
    current_weekday = context.current_dt.weekday()
    current_time_str = str(context.current_dt)[-8:]
    current_date = context.current_dt.date()

    # 判断当前时间段
    is_morning, is_afternoon, is_trading_time = get_trading_time_status(context)

    # 周一至周四14:50不买票的逻辑
    if current_weekday < 4 and is_afternoon:
#         log.info(f'当前为周{current_weekday + 1} 14:50，不执行买入')
        return

    # 交易记录会在买卖两端持续补充，方便盘后复盘
    if not hasattr(g, 'trade_records'):
        g.trade_records = {}

    # 初始化今日买入列表（存储字典：股票代码+买入前价格）
    if not hasattr(g, 'today_buy_list'):
        g.today_buy_list = []
    else:
        g.today_buy_list.clear()  # 清空之前的记录（确保只保留当日待买入信息）
    """买入时记录详细信息"""
    # 确保有 today_trades 列表记录当日交易
    if not hasattr(g, 'today_trades'):
        g.today_trades = []

    # 这些列表会在下面逐段填充，最后汇总进评分与排序流程
    lblt_stocks = []  # 连板龙头
    rzq_stocks = []  # 弱转强
    gk_stocks = []  # 一进二
    dk_stocks = []  # 首板低开
    fxsbdk_stocks = []  # 反向首板低开

    current_data = get_current_data()
    date_now = context.current_dt.strftime("%Y-%m-%d")
    date = transform_date(context.previous_date, 'str')

    if is_morning:
        # 1. 连板龙头（最高优先级）
        if g.lblt:
            # 全部连板股票
            ccd = get_continue_count_df(g.lblt, date, 20) if g.lblt else pd.DataFrame(index=[], data={'count': [],
                                                                                                      'extreme_count': []})

            # 重置索引，将股票代码作为普通列，便于安全访问
            if not ccd.empty:
                ccd_reset = ccd.reset_index().rename(columns={'index': 'code'})
                # 确保股票代码列为字符串
                ccd_reset['code'] = ccd_reset['code'].astype(str)
            else:
                ccd_reset = pd.DataFrame(columns=['code', 'count', 'extreme_count'])

            # 最高连板
            M = ccd['count'].max() if not ccd.empty else 0
            # 筛选龙头股票
            CCD = ccd[ccd['count'] == M] if M != 0 else pd.DataFrame(index=[], data={'count': [], 'extreme_count': []})
            lt = list(CCD.index)
            # 转换为字符串列表（确保类型一致）
            lt = [str(code) for code in lt]
            # 情绪
            emo = M
            g.emo_count.append(emo)
            # 周期
            cyc = g.emo_count[-1] if g.emo_count[-1] == max(g.emo_count[-3:]) and g.emo_count[-1] != 0 else 0
            cyc = 1 if cyc == emo else 0

            # 热门股票池
            try:
                dct = get_concept(g.lblt, date)
                hot_concept = get_hot_concept(dct, date)
                hot_stocks = filter_concept_stock(dct, hot_concept)
            except Exception as e:
#                 log.warning(f"获取热门概念失败: {e}")
                hot_stocks = []

            # 高风险股票筛选（直接读 get_stock_list() 中已缓存的结果，避免重复计算）
            high_risk_stocks = getattr(g, 'high_risk_stocks_today', [])
            if high_risk_stocks:
                # log.info(
                #     f"近5日有2个以上连续一字板的高风险个股(将被排除，来自缓存): {[get_security_info(s).display_name for s in high_risk_stocks]}")

            # 龙头特征筛选
                pass
            condition_dct = {}
            for s in lt:
                if s in high_risk_stocks:
                    continue
                if is_limit_up_open(s, current_data):
                    # log.info(f"早盘开盘涨停(将被排除): {s} {get_security_info(s).display_name}")
                    continue

                try:
                    stock_name = get_security_info(s).display_name

                    # 获取股票概念（确保为列表）
                    try:
                        concepts = get_concept([s], date).get(s, [])
                        if not isinstance(concepts, list):
                            concepts = list(concepts) if hasattr(concepts, '__iter__') else []
                        stock_concepts = concepts
                    except Exception as e:
#                         log.warning(f"获取{s}概念失败: {e}")
                        stock_concepts = []
                    hc = len(stock_concepts) > 0

                    # 独食：通过重置后的DataFrame安全获取
                    row = ccd_reset[ccd_reset['code'] == s]
                    if not row.empty:
                        ds = row.iloc[0]['extreme_count']
                    else:
                        ds = 0
                        # log.warning(f"股票{s}不在ccd_reset中，ds设为0")

                    # 市值
                    try:
                        sz_df = get_fundamentals(
                            query(valuation.code, valuation.circulating_market_cap).filter(valuation.code == s),
                            date
                        )
                        sz = sz_df.iloc[0, 1] if not sz_df.empty else 0
                    except Exception as e:
#                         log.warning(f"获取{s}市值失败: {e}")
                        sz = 0

                    # 换手
                    try:
                        hs_result = HSL([s], date)
                        hs = hs_result[0].get(s, 0) if hs_result and len(hs_result) > 0 else 0
                    except Exception as e:
#                         log.warning(f"获取{s}换手率失败: {e}")
                        hs = 0

                    # 龙头概念
                    c = 1 if s in hot_stocks else 0

                    # 逻辑判断
                    condition = ''
                    if hs < 35 and ds < 10 and emo >= 2:
                        if cyc == 1 and sz < 300:
                            condition += '上升周期'
                        if ds < 3 and 10 < hs < 25:
                            condition += ('+' if condition else '') + '资金接力'
                        if c == 1 and emo <= 6:
                            condition += (
                                             '+' if condition else '') + f'题材初期({",".join(stock_concepts[:3]) if hc else "未知"})'

                    if condition:
                        condition_dct[s] = f"{stock_name} —— {condition}"
                except Exception as e:
#                     log.error(f"龙头特征筛选{s}失败: {str(e)}")
                    continue

            stock_list = list(condition_dct.keys())
            # log.info(f"龙头股筛选结果(已排除高风险股票): {[get_security_info(s).display_name for s in stock_list]}")

            # 因子过滤
            df = get_factor_filter_df(context, stock_list, g.jqfactor, g.sort)
            lblt_stocks = list(df.index)

        # 2. 弱转强（第二优先级）
        # 批量获取估值数据，避免循环内逐只调用
        all_valuations_rzq = get_valuation(g.reversal, start_date=context.previous_date,
                                           end_date=context.previous_date,
                                           fields=['turnover_ratio', 'market_cap', 'circulating_market_cap']) \
            if g.reversal else pd.DataFrame()
        # 批量获取集合竞价数据
        all_auctions_rzq = get_call_auction(g.reversal, start_date=date_now, end_date=date_now,
                                            fields=['time', 'volume', 'current']) \
            if g.reversal else pd.DataFrame()
        if not all_auctions_rzq.empty:
            all_auctions_rzq = all_auctions_rzq.set_index('code')

        # 获取断板龙头列表
        broken_leaders = getattr(g, 'broken_leaders', [])

        for s in g.reversal:
            hist_data = attribute_history(s, 4, '1d', fields=['open', 'close', 'volume', 'money'], skip_paused=True)
            if len(hist_data) < 4:
                continue

            is_broken = s in broken_leaders

            # 1. 前三天涨幅（保持不变）
            increase_ratio = (hist_data['close'].iloc[-1] - hist_data['close'].iloc[0]) / hist_data['close'].iloc[0]
            if increase_ratio > 0.28:
                # log.debug(f"弱转强过滤 {s}：前三天涨幅{increase_ratio:.2%}>28%")
                continue

            # 2. 前一日收盘价与开盘价关系（保持不变）
            open_close_ratio = (hist_data['close'].iloc[-1] - hist_data['open'].iloc[-1]) / hist_data['open'].iloc[-1]
            if open_close_ratio < -0.05:
                # log.debug(f"弱转强过滤 {s}：前一日跌幅{open_close_ratio:.2%}<-5%")
                continue

            # 3. 成交额与均价涨幅（对断板龙头放宽）
            prev_money = hist_data['money'].iloc[-1]
            prev_volume = hist_data['volume'].iloc[-1]
            prev_close = hist_data['close'].iloc[-1]
            avg_price_increase = prev_money / prev_volume / prev_close - 1

            if is_broken:
                money_min = 2e8  # 断板龙头成交额下限2亿
                money_max = 50e8  # 断板龙头成交额上限50亿
                avg_min = -0.06  # 均价涨幅下限-6%
            else:
                money_min = 3e8
                money_max = 19e8
                avg_min = -0.04

            if avg_price_increase < avg_min or prev_money < money_min or prev_money > money_max:
                # log.debug(
                #     f"弱转强过滤 {s}：成交额或均价涨幅不符合 (is_broken={is_broken}, money={prev_money / 1e8:.2f}亿, avg={avg_price_increase:.2%})")
                continue

            # 4. 市值（对断板龙头放宽）
            if not all_valuations_rzq.empty:
                val_row = all_valuations_rzq[all_valuations_rzq['code'] == s]
            else:
                val_row = pd.DataFrame()
            if val_row.empty:
#                 log.debug(f"弱转强过滤 {s}：无估值数据")
                continue
            market_cap = val_row['market_cap'].iloc[0]
            circ_market_cap = val_row['circulating_market_cap'].iloc[0]

            if is_broken:
                cap_ok = (market_cap >= 50 and circ_market_cap <= 600)  # 断板龙头总市值≥50亿，流通市值≤600亿
            else:
                cap_ok = (market_cap >= 70 and circ_market_cap <= 520)  # 普通要求

            if not cap_ok:
                # log.debug(
                #     f"弱转强过滤 {s}：市值不符合 (is_broken={is_broken}, market_cap={market_cap}亿, circ_market_cap={circ_market_cap}亿)")
                continue

            # 5. 量能条件（保持不变，如有需要也可放宽）
                pass
            if rise_low_volume(s, context):
                # log.debug(f"弱转强过滤 {s}：量能条件 rise_low_volume 触发")
                continue

            # 6. 集合竞价（对断板龙头放宽竞价量占比和开盘比值）
            if not all_auctions_rzq.empty and s in all_auctions_rzq.index:
                auction_row = all_auctions_rzq.loc[s]
                if isinstance(auction_row, pd.DataFrame):
                    _volume = auction_row['volume'].iloc[0]
                    _current = auction_row['current'].iloc[0]
                else:
                    _volume = auction_row['volume']
                    _current = auction_row['current']

                vol_ratio = _volume / prev_volume
                min_vol_ratio = 0.01 if is_broken else 0.03  # 断板龙头允许1%
                if vol_ratio < min_vol_ratio:
                    # log.debug(f"弱转强过滤 {s}：竞价量占比{vol_ratio:.2%}<{min_vol_ratio:.0%}")
                    continue

                current_ratio = _current / (current_data[s].high_limit / 1.1)
                if is_broken:
                    ratio_low = 0.95
                    ratio_high = 1.10
                else:
                    ratio_low = 0.98
                    ratio_high = 1.09

                if current_ratio < ratio_low or current_ratio > ratio_high:
                    # log.debug(f"弱转强过滤 {s}：开盘比值{current_ratio:.2f}不在[{ratio_low:.2f},{ratio_high:.2f}]")
                    continue
            else:
                # log.debug(f"弱转强过滤 {s}：无集合竞价数据")
                continue

            # 7. 主线分（保持不变）
            mainline_score = calculate_mainline_score_optimized(s, context)
            if mainline_score == 0:
                # log.info(f"弱转强个股 {s} 主线分为0，不纳入买入候选")
                continue

            # 新增主线分上限判断
            max_mainline = g.dynamic_params.get('rzq_max_mainline', 6)
            if mainline_score > max_mainline:
                # log.info(f"弱转强个股 {s} 主线分{mainline_score}>{max_mainline}，不纳入买入候选")
                continue

            rzq_stocks.append(s)

        # 3. 一进二（第三优先级）
        # 批量获取估值和集合竞价数据，避免循环内逐只调用
        all_valuations_gk = get_valuation(g.gap_up, start_date=context.previous_date,
                                          end_date=context.previous_date,
                                          fields=['turnover_ratio', 'market_cap', 'circulating_market_cap']) \
            if g.gap_up else pd.DataFrame()
        all_auctions_gk = get_call_auction(g.gap_up, start_date=date_now, end_date=date_now,
                                           fields=['time', 'volume', 'current']) \
            if g.gap_up else pd.DataFrame()
        if not all_auctions_gk.empty:
            all_auctions_gk = all_auctions_gk.set_index('code')

        for s in g.gap_up:
            # 条件一：均价，金额，市值，换手率
            prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
            if prev_day_data.empty:
                continue
            avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][
                0] * 1.1 - 1
            if avg_price_increase_value < 0.07 or prev_day_data['money'][0] < 5.5e8 or prev_day_data['money'][0] > 20e8:
                continue

            # 市值条件
            if not all_valuations_gk.empty:
                val_row_gk = all_valuations_gk[all_valuations_gk['code'] == s]
            else:
                val_row_gk = pd.DataFrame()
            if val_row_gk.empty or val_row_gk['market_cap'].iloc[0] < 50 or \
                    val_row_gk['circulating_market_cap'].iloc[0] > 520:
                pass
                continue

            # 条件二：左压
            if rise_low_volume(s, context):
                continue

            # 条件三：集合竞价
            if all_auctions_gk.empty or s not in all_auctions_gk.index:
                continue
            auction_row_gk = all_auctions_gk.loc[s]
            if isinstance(auction_row_gk, pd.DataFrame):
                gk_vol = auction_row_gk['volume'].iloc[0]
                gk_current = auction_row_gk['current'].iloc[0]
            else:
                gk_vol = auction_row_gk['volume']
                gk_current = auction_row_gk['current']
            if gk_vol / prev_day_data['volume'][-1] < 0.03:
                continue
            current_ratio = gk_current / (current_data[s].high_limit / 1.1)
            if current_ratio <= 1 or current_ratio >= 1.06:
                continue

            # 条件4：价格<500
            current_price = current_data[s].last_price
            if current_price > 500:
                continue

            

            # 条件5：量比范围检查
            try:
                last_volume, last_2_volume, volume_ratio = get_volume_data(s, context)
                if volume_ratio < 1.15 or volume_ratio > 6.58:
                    continue
            except Exception as e:
                # log.warning(f"获取股票 {s} 的量比数据失败: {str(e)}")
                continue

            # 如果股票满足所有条件，则添加到列表中
            gk_stocks.append(s)

        # 4. 首板低开（第四优先级）
        if g.gap_down:
            stock_list = g.gap_down
            # 计算相对位置
            rpd = get_relative_position_df(stock_list, date, 60)
            rpd = rpd[rpd['rp'] <= 0.5]
            stock_list = list(rpd.index)

            # 低开筛选
            df = get_price(stock_list, end_date=date, frequency='daily', fields=['close'], count=1, panel=False,
                           fill_paused=False, skip_paused=True).set_index('code') if len(
                stock_list) != 0 else pd.DataFrame()
            if not df.empty:
                df['open_pct'] = [current_data[s].day_open / df.loc[s, 'close'] for s in stock_list]
                df = df[(0.955 <= df['open_pct']) & (df['open_pct'] <= 0.97)]  # 筛选3个点左右低开
                stock_list = list(df.index)

                for s in stock_list:
                    prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
                    if prev_day_data['money'][0] >= 1e8:
                        dk_stocks.append(s)

        # 5. 反向首板低开（第五优先级）
        if g.fxsbdk:
            # 获取非连板涨停的股票
            ccd = get_continue_count_df_ll(g.fxsbdk, date, 10)
            lb_list = list(ccd.index)
            stock_list = [s for s in g.fxsbdk if s not in lb_list]

            # 计算相对位置
            rpd = get_relative_position_df(stock_list, date, 60)
            rpd = rpd[rpd['rp'] <= 0.5]
            stock_list = list(rpd.index)

            # 低开筛选
            df = get_price(stock_list, end_date=date, frequency='daily', fields=['close'], count=1, panel=False,
                           fill_paused=False, skip_paused=True).set_index('code') if len(
                stock_list) != 0 else pd.DataFrame()
            if not df.empty:
                df['open_pct'] = [current_data[s].day_open / df.loc[s, 'close'] for s in stock_list]
                df = df[(1.04 <= df['open_pct']) & (df['open_pct'] < 1.10)]  # 筛选特定低开幅度
                fxsbdk_stocks = list(df.index)
                
        # ========== 新增：市盈率过滤（仅4月生效） ==========
        lblt_stocks = filter_by_valuation(lblt_stocks, context)
        rzq_stocks = filter_by_valuation(rzq_stocks, context)
        gk_stocks = filter_by_valuation(gk_stocks, context)
        dk_stocks = filter_by_valuation(dk_stocks, context)
        fxsbdk_stocks = filter_by_valuation(fxsbdk_stocks, context)

        # 关键：将筛选后的列表赋值为全局变量，供get_buy_reason函数调用
        g.lblt_stocks = lblt_stocks
        g.rzq_stocks = rzq_stocks
        g.gk_stocks = gk_stocks
        g.dk_stocks = dk_stocks
        g.fxsbdk_stocks = fxsbdk_stocks

        # 按优先级合并股票列表（去重，保留首次出现顺序）
        qualified_stocks = []
        # 创建优先级列表映射，便于根据配置动态调整顺序
        priority_lists = {
            "lb": lblt_stocks,  # 连板龙头
            "yje": gk_stocks,  # 一进二
            "rzq": rzq_stocks,  # 弱转强
            "dk": dk_stocks,  # 首板低开
            "fxsbdk": fxsbdk_stocks  # 反向首板低开
        }

        # 按优先级配置合并股票列表
        seen = set()
        for priority_type in g.priority_config:
            stock_list = priority_lists.get(priority_type, [])
            for s in stock_list:
                if s not in seen:
                    seen.add(s)
                    qualified_stocks.append(s)


        # 评分筛选：降低最低评分要求
        qualified_stocks = filter_stocks_by_score_optimized(
            qualified_stocks,
            context,
            min_score=g.min_score,
            max_stocks=100
        )
        g.qualified_stocks = qualified_stocks
        if not qualified_stocks:
            log.info("没有符合条件的股票")
            log.info("选股结果：0只，无")
        if not qualified_stocks:
#             log.info("没有符合条件的股票")
            send_message('今日无目标个股')
#             log.info('选股结果：0只，无')
            return

        # 根据qualified_stocks过滤各个模式个股，只保留同时存在于qualified_stocks中的股票
        qualified_set = set(qualified_stocks)  # 转为集合加速查找
        g.lblt_stocks = [s for s in g.lblt_stocks if s in qualified_set]
        g.rzq_stocks = [s for s in g.rzq_stocks if s in qualified_set]
        g.gk_stocks = [s for s in g.gk_stocks if s in qualified_set]
        g.dk_stocks = [s for s in g.dk_stocks if s in qualified_set]
        g.fxsbdk_stocks = [s for s in g.fxsbdk_stocks if s in qualified_set]
        lblt_stocks = g.lblt_stocks
        rzq_stocks = g.rzq_stocks
        gk_stocks = g.gk_stocks
        dk_stocks = g.dk_stocks
        fxsbdk_stocks = g.fxsbdk_stocks

    # 非周五交易日处理
    if current_weekday < 4:
        qualified_stock_text = format_stock_list_with_names(qualified_stocks)
        log.info(f"选股结果：{len(qualified_stocks)}只，{qualified_stock_text}")
        send_message('今日选股：' + ','.join(qualified_stocks))

    # 周五特殊处理逻辑
    if current_weekday == 4:  # 周五
        # 早盘9:28:10执行：先筛选股票，再买入连板龙头和首板低开
        if is_morning:
            log.info("周五早盘：筛选股票并买入连板龙头和首板低开")
            log.info(f"周五早盘选出的个股：{g.qualified_stocks}")
            # 保存选股结果到全局变量（已在之前设置）
#             log.info("周五早盘：筛选股票并买入连板龙头和首板低开")
#             log.info(f"周五早盘选出的个股：{g.qualified_stocks}")

            # 提取连板龙头和首板低开股票
            friday_morning_stocks = []
            # 连板龙头
            if hasattr(g, 'lblt_stocks') and g.lblt_stocks:
                current_data = get_current_data()
                filtered_lblt_stocks = []
                for s in g.lblt_stocks:
                    if s in g.qualified_stocks:
                        # 检查开盘价格是否超过8%
                        try:
                            stock_data = current_data[s]
                            # log.debug(f"检查连板龙头 {s} 的开盘价格，stock_data属性: {dir(stock_data)}")
                            open_price = stock_data.day_open
                            # 检查day_open是否有效（天回测时可能为NaN）
                            if open_price is None or (isinstance(open_price, float) and math.isnan(open_price)):
#                                 log.error(f"连板龙头 {s} 的开盘价无效")
                                filtered_lblt_stocks.append(s)  # 出错时保留股票
                            else:
                                # 获取前收盘价，使用attribute_history
#                                 log.debug(f"获取连板龙头 {s} 的前收盘价")
                                prev_data = attribute_history(s, 1, '1d', fields=['close'], skip_paused=True)
#                                 log.debug(f"prev_data形状: {prev_data.shape}, 内容: {prev_data}")
                                if not prev_data.empty:
                                    prev_close = prev_data['close'].iloc[0]
                                    if prev_close > 0:
                                        open_change_pct = (open_price - prev_close) / prev_close * 100
                                        if open_change_pct <= 8:
                                            filtered_lblt_stocks.append(s)
                                        else:
#                                             log.info(f"连板龙头 {s} 开盘涨幅{open_change_pct:.2f}%超过8%，筛除")
                                            pass
                                else:
#                                     log.error(f"无法获取连板龙头 {s} 的前收盘价")
                                    filtered_lblt_stocks.append(s)  # 出错时保留股票
                        except Exception as e:
#                             log.error(f"检查连板龙头 {s} 开盘价格时出错: {e}")
                            import traceback
#                             log.error(f"错误堆栈: {traceback.format_exc()}")
                            filtered_lblt_stocks.append(s)  # 出错时保留股票
                friday_morning_stocks.extend(filtered_lblt_stocks)
            # 首板低开
            if hasattr(g, 'dk_stocks') and g.dk_stocks:
                friday_morning_stocks.extend([s for s in g.dk_stocks if s in g.qualified_stocks])

            # 去重
            friday_morning_stocks = list(set(friday_morning_stocks))

            if friday_morning_stocks:
                log.info(f"周五早盘候选股票（连板龙头+首板低开）：{friday_morning_stocks}")
                log.info("周五早盘股票已存入候选池，09:33内外比过滤后买入")
#                 log.info(f"周五早盘候选股票（连板龙头+首板低开）：{friday_morning_stocks}")
                # 不在此买入——早盘买入统一在09:33由buy_after_auction_filter执行内外比过滤后买入
#                 log.info("周五早盘股票已存入候选池，09:33内外比过滤后买入")
                pass
            else:
                log.info("周五早盘无可买入的连板龙头或首板低开股票")
#                 log.info("周五早盘无可买入的连板龙头或首板低开股票")

                pass
            return

        # 14:50执行：买入其他模式（弱转强、一进二、反向首板低开）
        if is_afternoon:
            log.info("周五14:50执行建仓（弱转强、一进二、反向首板低开）")
#             log.info("周五14:50执行建仓（弱转强、一进二、反向首板低开）")
            if not hasattr(g, 'qualified_stocks') or not g.qualified_stocks:
#                 log.warning("周五14:50：未找到可交易股票")
                return

            # 早盘筛选结果
            qualified_stocks = g.qualified_stocks
            lblt_stocks = g.lblt_stocks
            rzq_stocks = g.rzq_stocks
            gk_stocks = g.gk_stocks
            dk_stocks = g.dk_stocks
            fxsbdk_stocks = g.fxsbdk_stocks

            # 提取非连板龙头和非首板低开的其他模式股票（弱转强、一进二、反向首板低开）
            other_mode_stocks = []
            # 弱转强
            if hasattr(g, 'rzq_stocks') and g.rzq_stocks:
                other_mode_stocks.extend([s for s in g.rzq_stocks if s in qualified_stocks])
            # 一进二
            if hasattr(g, 'gk_stocks') and g.gk_stocks:
                other_mode_stocks.extend([s for s in g.gk_stocks if s in qualified_stocks])
            # 反向首板低开
            if hasattr(g, 'fxsbdk_stocks') and g.fxsbdk_stocks:
                other_mode_stocks.extend([s for s in g.fxsbdk_stocks if s in qualified_stocks])

            # 去重
            other_mode_stocks = list(set(other_mode_stocks))

            if not other_mode_stocks:
#                 log.info("周五14:50：无其他模式股票")
                send_message('周五下午无目标个股')
                return

            # 二次筛选（仅对非连板龙头和非首板低开的股票进行优化筛选）
            filtered_other_stocks = optimize_friday_trading_logic(context, other_mode_stocks)

            if not filtered_other_stocks:
#                 log.info("周五14:50：优化筛选后无符合条件的其他模式股票")
                send_message('周五下午无目标个股')
                return

            # 更新全局变量，只保留下午买入的股票
            g.qualified_stocks = filtered_other_stocks
            qualified_set = set(filtered_other_stocks)
            # 更新各模式列表（只保留下午买入的股票）
            g.rzq_stocks = [s for s in rzq_stocks if s in qualified_set]
            g.gk_stocks = [s for s in gk_stocks if s in qualified_set]
            g.fxsbdk_stocks = [s for s in fxsbdk_stocks if s in qualified_set]
            # 注意：连板龙头和首板低开在上午已买，这里不再包含
            g.lblt_stocks = []  # 清空，避免重复
            g.dk_stocks = []  # 清空

            send_message('今日下午选股（其他模式）：' + ','.join(g.qualified_stocks))
#             log.info(f"选股结果：{len(g.qualified_stocks)}只，{','.join(g.qualified_stocks)}")

    # ==================== 买入执行逻辑 ====================  
    # 筛选连板龙头：开盘价格超过8%就筛除
    if hasattr(g, 'lblt_stocks') and g.lblt_stocks:
        current_data = get_current_data()
        filtered_lblt_stocks = []
        for s in g.lblt_stocks:
            # 检查开盘价格是否超过8%
            try:
                stock_data = current_data[s]
#                 log.debug(f"检查连板龙头 {s} 的开盘价格，stock_data属性: {dir(stock_data)}")
                open_price = stock_data.day_open
                # 检查day_open是否有效（天回测时可能为NaN）
                if open_price is None or (isinstance(open_price, float) and math.isnan(open_price)):
#                     log.error(f"连板龙头 {s} 的开盘价无效")
                    filtered_lblt_stocks.append(s)  # 出错时保留股票
                else:
                    # 获取前收盘价，使用attribute_history
#                     log.debug(f"获取连板龙头 {s} 的前收盘价")
                    prev_data = attribute_history(s, 1, '1d', fields=['close'], skip_paused=True)
#                     log.debug(f"prev_data形状: {prev_data.shape}, 内容: {prev_data}")
                    if not prev_data.empty:
                        prev_close = prev_data['close'].iloc[0]
                        if prev_close > 0:
                            open_change_pct = (open_price - prev_close) / prev_close * 100
                            if open_change_pct <= 8:
                                filtered_lblt_stocks.append(s)
                            else:
#                                 log.info(f"连板龙头 {s} 开盘涨幅{open_change_pct:.2f}%超过8%，筛除")
                                pass
                    else:
#                         log.error(f"无法获取连板龙头 {s} 的前收盘价")
                        filtered_lblt_stocks.append(s)  # 出错时保留股票
            except Exception as e:
#                 log.error(f"检查连板龙头 {s} 开盘价格时出错: {e}")
                import traceback
#                 log.error(f"错误堆栈: {traceback.format_exc()}")
                filtered_lblt_stocks.append(s)  # 出错时保留股票
        g.lblt_stocks = filtered_lblt_stocks
        # 同时从qualified_stocks中移除筛除的股票
        if hasattr(g, 'qualified_stocks'):
            g.qualified_stocks = [s for s in g.qualified_stocks if s not in g.lblt_stocks or s in filtered_lblt_stocks]
    
    # ==================== 买入执行逻辑 ====================
    # 早盘（09:26~09:28）不在此买入，统一在09:33由buy_after_auction_filter执行内外比过滤
    # 仅下午（周五14:51建仓）在此直接买入
    if is_afternoon:
        execute_buy(context, True)


def classify_broken_type(open_price, close_price, high_price, low_price, high_limit, change_pct):
    """
    判断断板属于哪一类。

    目的不是为了“好看地命名”，而是为了区分强弱：
    - 有些断板说明资金还很强，只是没封住；
    - 有些断板说明已经明显走弱，后续参与价值不高。

    参数：
        open_price: 开盘价
        close_price: 收盘价
        high_price: 最高价
        low_price: 最低价
        high_limit: 涨停价
        change_pct: 涨跌幅（百分比）

    返回：
        str: 断板类型
    """
    try:
        # 判断是否触及涨停价（允许0.1%误差）
        touched_limit = (high_price >= high_limit * 0.999)

        # 判断开盘是否涨停
        is_high_open = (open_price >= high_limit * 0.999)

        # 1. 一字板炸板（开盘涨停，盘中炸板）
        if is_high_open and touched_limit:
            if close_price < high_limit * 0.99:  # 收盘未封板
                if change_pct > 5:
                    return "一字炸板-强势"
                elif change_pct > 0:
                    return "一字炸板-弱势"
                else:
                    return "一字炸板-跌破"

        # 2. 冲高回落（盘中触及涨停，但未封住）
        if not is_high_open and touched_limit:
            if close_price < high_limit * 0.99:
                if change_pct > 5:
                    return "冲高回落-强势"
                elif change_pct > 0:
                    return "冲高回落-弱势"
                else:
                    return "冲高回落-跌破"

        # 3. 高开低走（高开但未触及涨停）
        if not touched_limit and open_price > close_price:
            if change_pct > 3:
                return "高开低走-小幅回调"
            elif change_pct > 0:
                return "高开低走-大幅回调"
            else:
                return "高开低走-翻绿"

        # 4. 低开反抽（低开后反弹）
        if open_price < close_price:
            if change_pct > 5:
                return "低开反抽-强势"
            elif change_pct > 0:
                return "低开反抽-弱势"
            else:
                return "低开反抽-失败"

        # 5. 直接跌停
        if change_pct < -8:
            return "直接跌停"

        # 6. 大幅下跌
        if change_pct < -5:
            return "大幅下跌"

        # 7. 其他情况
        if change_pct > 0:
            return "普通断板-小涨"
        elif change_pct < 0:
            return "普通断板-小跌"
        else:
            return "普通断板-平盘"

    except Exception as e:
#         log.error(f"⚠️ 断板类型分类失败: {e}")
        return "未知类型"


from datetime import datetime  # 只导入 datetime 类，不导入 time


def check_volume_for_buy(stock, context, current_data):
    """
    买入前的量能判断（简化版）
    条件：当日累计成交量 <= 昨日成交量 * 1.15
    说明：直接使用实时累计成交量，不再估算全天，避免早盘放大误差

    小白可以这样理解：
    这一步是在防“突然爆量但价格未必健康”的票。
    如果今天到当前时刻的成交量已经远超昨天，可能是情绪过热，也可能是出货，
    所以这里先保守一点，不急着追。
    """
    try:
        # 1. 基础数据检查
        if stock not in current_data or current_data[stock].paused:
            return False
        if current_data[stock].last_price == 0:
            return False

        # 2. 获取昨日成交量（股）
        hist = attribute_history(stock, 2, '1d', ['volume'], skip_paused=True)
        if len(hist) < 2 or hist['volume'].iloc[-2] == 0:
            return True  # 数据不足，保守允许买入
        yesterday_volume = hist['volume'].iloc[-2]

        # 3. 获取当日累计成交量（股）
        # 从开盘到当前时刻的分钟数据
        today_start = context.current_dt.replace(hour=9, minute=30, second=0)
        if context.current_dt < today_start:
            return True

        minute_data = get_price(stock, start_date=today_start, end_date=context.current_dt,
                                frequency='1m', fields=['volume'], skip_paused=True)
        if minute_data.empty:
            return True

        current_volume = minute_data['volume'].sum()

        # 4. 计算量能比例（当日累计 / 昨日）
        volume_ratio = current_volume / yesterday_volume

        # 5. 判断条件：当日累计量不超过昨日的1.15倍（即放量不超过15%）
        can_buy = volume_ratio <= 1.15

        # 6. 日志输出（单位统一为万股）
        stock_name = get_security_info(stock).display_name
#         log.info(f"📊 买入量能检测 {stock}({stock_name}):")
#         log.info(f"   时间: {context.current_dt.strftime('%H:%M')}")
#         log.info(f"   当日累计成交量: {current_volume / 10000:.2f}万股")
#         log.info(f"   昨日成交量: {yesterday_volume / 10000:.2f}万股")
#         log.info(f"   量能比例(当日/昨日): {volume_ratio:.2f}倍")
#         log.info(f"   买入条件(≤1.15): {'✅通过' if can_buy else '❌不通过'}")

        return can_buy

    except Exception as e:
#         log.error(f"买入量能检测出错 {stock}: {str(e)}")
        return False


def get_dynamic_correction(stock, minutes_elapsed, context):
    """
    动态修正因子：基于历史早盘成交量占比
    """
    default_factor = 0.8 + (minutes_elapsed / 120) * 0.4
    if minutes_elapsed <= 0:
        return default_factor

    try:
        # 获取过去20个交易日（不包括今天）
        trade_days = get_trade_days(end_date=context.current_dt.date(), count=21)
        trade_days = trade_days[:-1]
        if len(trade_days) < 5:
            return default_factor

        ratios = []
        for day in trade_days[-20:]:
            # 使用 datetime 直接构造，避免导入 time 对象
            start = datetime(day.year, day.month, day.day, 9, 30)
            end = datetime(day.year, day.month, day.day, 15, 0)
            day_minutes = get_price(stock, start_date=start, end_date=end,
                                    frequency='1m', fields=['volume'], skip_paused=True)
            if day_minutes.empty:
                continue

            total_vol = day_minutes['volume'].sum()
            if total_vol == 0:
                continue

            early_vol = day_minutes['volume'].iloc[:minutes_elapsed].sum()
            ratio = early_vol / total_vol
            ratios.append(ratio)

        if not ratios:
            return default_factor

        avg_early_ratio = sum(ratios) / len(ratios)
        if avg_early_ratio > 0:
            dynamic_factor = 1.0 / avg_early_ratio
            dynamic_factor = max(0.5, min(2.0, dynamic_factor))
            return dynamic_factor
        else:
            return default_factor

    except Exception as e:
#         log.warning(f"获取动态修正因子失败 {stock}: {e}，使用默认")
        return default_factor


def is_above_ma5(stock, context):
    try:
        df = attribute_history(stock, 6, '1d', ['close'], skip_paused=True, df=False)
        if df is None or len(df['close']) < 5:
            return False
        closes = df['close']
        ma5 = sum(closes[-5:]) / 5
        yesterday_close = closes[-1]
        return yesterday_close > ma5
    except Exception as e:
#         log.debug(f"is_above_ma5 检查 {stock} 失败: {e}")
        return False


def execute_buy(context, isFiltered=False, custom_stocks=None):
    """
    公共买入执行函数。

    这里才是真正负责“下单”的地方。

    参数:
        isFiltered:
            True  表示由 buy() 主动触发，此时更像正式建仓。
            False 表示由定时任务触发，此时会额外检查实时技术信号。
        custom_stocks:
            可选的临时股票池；不传时默认使用 g.qualified_stocks。
    """
    # 空仓检查
    if getattr(g, 'is_empty', False):
#         log.info("当前为空仓状态，不执行买入")
        return

    # 防御性初始化
    if not hasattr(g, 'stocks_limit_up_today'):
        g.stocks_limit_up_today = set()

    # 周五跳过定时买入任务（isFiltered=False），但保留 buy 函数调用的建仓（isFiltered=True）
    if context.current_dt.weekday() == 4 and not isFiltered:
        return

    # 确定本次下单的股票池来源
    if custom_stocks is not None:
        candidate_pool = custom_stocks
    else:
        if not g.qualified_stocks:
            # log.debug("合格买入股票列表为空，跳过公共买入逻辑")
            return
        candidate_pool = g.qualified_stocks

    # 1. 校验合格股票列表是否为空
    if not candidate_pool:
        # log.debug("合格买入股票列表为空，跳过公共买入逻辑")
        return

    current_data = get_current_data()
    current_positions = len(context.portfolio.positions)
    available_positions = g.position_limit - current_positions

    # 判断当前时间段
    is_morning, is_afternoon, is_trading_time = get_trading_time_status(context)

    if available_positions <= 0:
        # log.debug(f"已达最大持仓限制{g.position_limit}，不执行买入")
        return

    # 检查持股仓位比例，超过85%不买入
    total_value = context.portfolio.total_value
    positions_value = context.portfolio.positions_value
    position_ratio = positions_value / total_value if total_value > 0 else 0
    if position_ratio >= 0.85:
#         log.debug(f"持股仓位已达{position_ratio:.2%}，超过85%阈值，跳过买入")
        return

    # 这里做“下单前最后一轮过滤”，只保留当前这一刻依然值得买的票
    current_buy_candidates = []
    # 使用原始的合格股票池（早盘筛选结果），不修改全局变量
    for s in candidate_pool:
        # ----- 1. 涨停过滤（统一处理）-----
        if s in g.stocks_limit_up_today:
#             log.info(f"❌ {s} 当天曾涨停，跳过买入")
            continue

        stock_data = current_data[s]
        current_price = stock_data.last_price
        high_limit = stock_data.high_limit

        # 当前涨停检测（防止未及时标记）
        if current_price >= high_limit * 0.995:
            g.stocks_limit_up_today.add(s)
#             log.info(f"❌ {s} 当前价格涨停，跳过买入")
            continue

        # ----- 2. 技术指标检查（仅定时任务）-----
        # 盘中定时任务需要再检查一次实时信号，
        # 避免早盘入选、盘中走坏的股票被误买。
        if not isFiltered:  # 定时任务需要实时技术指标
            # 同花顺指标判断
            ths_signals = calculate_ths_indicators(s, context, 120, '5m')
            if ths_signals['trend_color'] != 'red':
                continue
            if ths_signals['last_signal_type'] == '波段卖' and ths_signals['last_signal_offset'] <= 2:
                continue
            if not is_above_ma5(s, context):
                continue

            # 量能检测
            if not check_volume_for_buy(s, context, current_data):
                continue

        # 所有条件满足，加入本次候选
        current_buy_candidates.append(s)

    # 如果没有候选，直接返回
    if not current_buy_candidates:
#         log.info("无符合条件的未持仓股票，跳过买入")
        return

    # ========== 排序：按模式优先级和评分 ==========
    sorted_stocks = sort_stocks_by_priority(
        current_buy_candidates,
        g.lblt_stocks,
        g.gk_stocks,
        g.rzq_stocks,
        g.dk_stocks,
        g.fxsbdk_stocks
    )

    # ========== 过滤已持仓的股票 ==========
    holding_stocks = set(context.portfolio.positions.keys())
    candidate_stocks = [s for s in sorted_stocks if s not in holding_stocks]
    if not candidate_stocks:
#         log.info("过滤后无符合条件的未持仓股票，跳过买入")
        return

    # 计算可买入数量
    buy_count = min(len(candidate_stocks), available_positions)
    if buy_count <= 0:
#         log.info("无可买入股票或仓位已满")
        return

    # 资金分配思路：
    # 先按可用仓位平分，再和当前可用现金比较，取更保守的金额。
    # 用聚宽自己的总资产等分，不感知QMT充值；total_value只会因交易盈亏变化
    # 保留cash_reserve_ratio作为缓冲，防止QMT执行时价格小幅上涨不够买
    target_per_position = context.portfolio.total_value * (1 - g.cash_reserve_ratio) / buy_count
    value = min(target_per_position, context.portfolio.available_cash / buy_count)
    log.info(f"买入资金：{value:.2f}，目标单仓：{target_per_position:.2f}，总资产：{context.portfolio.total_value:.2f}")
#     log.info(f"买入资金：{value:.2f}，目标单仓：{target_per_position:.2f}，总资产：{context.portfolio.total_value:.2f}")

    # 执行买入
    bought_count = 0
    for s in candidate_stocks[:buy_count]:
        price = current_data[s].last_price
        if context.portfolio.available_cash < price * 100:
            continue

        current_time = context.current_dt.strftime('%H:%M:%S')
        reason = get_buy_reason(s, context)

        buy_quantity = int(value / price / 100) * 100
        if buy_quantity <= 0:
            continue

        last_volume, last_2_volume, trade_volume_ra = get_volume_data(s, context)

        trade_info = {
            'time': current_time,
            'stock': s,
            'action': '买入',
            'price': price,
            'buy_quantity': buy_quantity,
            'reason': reason,
            'market_value': context.portfolio.total_value,
            'last_volume': float(last_volume) if last_volume is not None else 0,
            'last_2_volume': float(last_2_volume) if last_2_volume is not None else 0,
            'trade_volume_ra': float(trade_volume_ra) if trade_volume_ra is not None else None,
            'sell_date': None,
            'sell_price': None,
            'sell_time': None,
            'sell_quantity': 0,
            'sell_value': 0,
            'profit_pct': None
        }

        try:
            order_style = MarketOrderStyle(price)
            order_result = order_value(s, value, order_style)
            if order_result:
                actual_quantity = order_result.amount
                trade_info['buy_quantity'] = actual_quantity

                if not hasattr(g, 'today_trades'):
                    g.today_trades = []
                g.today_trades.append(trade_info)
                bought_count += 1
                stock_name = current_data[s].name if s in current_data and hasattr(current_data[s], 'name') else s
                log.info(f"买入执行: {s}({stock_name}), 时间={current_time}, 价格={price:.2f}, 金额={value:.2f}, 数量={actual_quantity}, 原因={trade_info['reason']}")

#                 log.info(f"\n==== 买入执行 {s} ====")
#                 log.info(f"时间: {current_time}")
#                 log.info(f"买入价格: {price:.2f}")
#                 log.info(f"买入金额: {value:.2f}")
#                 log.info(f"买入数量: {actual_quantity}")
#                 log.info(f"买入原因: {trade_info['reason']}")
#                 log.info(f"当前总值: {trade_info['market_value']:.2f}")
#                 log.info(
#                     f"昨日量: {last_volume}  前一日量: {last_2_volume}  量能比: {trade_volume_ra if trade_volume_ra is not None else 'NA'}")
#                 log.info("————————————————————")

                send_message(f'买入 {s} 价格:{price:.2f} 数量:{actual_quantity}')
            else:
#                 log.error(f"买入 {s} 失败")
                pass
        except Exception as e:
#             log.error(f"买入 {s} 时发生错误: {str(e)}")

            pass
    if bought_count == 0:
        log.info("本次未买入任何股票")
#         log.info("本次未买入任何股票")
        send_message('本次未买入任何股票')


def sort_stocks_by_priority(qualified_stocks, lblt_stocks, gk_stocks, rzq_stocks, dk_stocks, fxsbdk_stocks):
    """
    按“策略模式优先级 + 个股评分”双重规则排序。

    简单理解：
    - 先看股票属于哪个模式；
    - 模式之间按 g.priority_config 排序；
    - 同一个模式里，再按评分高低排序。

    参数:
        qualified_stocks: 符合条件的股票列表
        lblt_stocks: 连板龙头股票列表
        gk_stocks: 一进二股票列表
        rzq_stocks: 弱转强股票列表
        dk_stocks: 首板低开股票列表
        fxsbdk_stocks: 反向首板低开股票列表

    返回:
        排序后的股票列表
    """
    try:
        # ========== 1. 创建模式优先级映射 ==========
        pattern_priority_map = {}
        for i, pattern in enumerate(g.priority_config):
            # 优先级：第一个模式优先级最高
            pattern_priority_map[pattern] = len(g.priority_config) - i

#         log.info(f"📊 模式优先级映射: {pattern_priority_map}")
#         log.info(f"📋 当前策略优先级: {' > '.join(g.priority_config)}")

        # ========== 2. 创建股票到模式的映射（按优先级顺序） ==========
        # 一只股票可能同时落入多个模式。
        # 这里让“高优先级模式”覆盖低优先级模式。
        stock_pattern_map = {}

        # 定义模式到股票列表的映射
        pattern_stocks_map = {
            "lb": lblt_stocks,
            "yje": gk_stocks,
            "rzq": rzq_stocks,
            "dk": dk_stocks,
            "fxsbdk": fxsbdk_stocks
        }
        # 按照优先级从低到高的顺序映射（后面的会覆盖前面的）
        # 这样优先级高的模式会保留
        for pattern in reversed(g.priority_config):
            stocks = pattern_stocks_map.get(pattern, [])
            for stock in stocks:
                if stock in qualified_stocks:  # 只映射在候选列表中的股票
                    stock_pattern_map[stock] = pattern

        qualified_stocks = [s for s in qualified_stocks if s in stock_pattern_map]

        # ========== 3. 为每个股票创建排序信息 ==========
        stock_sort_info = []
        for stock in qualified_stocks:
            # 获取评分
            score = 0
            if hasattr(g, 'score_cache') and stock in g.score_cache:
                score = g.score_cache[stock].get('total_score', 0)

            # 获取模式和优先级
            pattern = stock_pattern_map.get(stock, None)
            priority = pattern_priority_map.get(pattern, 0) if pattern else 0

            # 添加到排序列表
            stock_sort_info.append({
                'stock': stock,
                'pattern': pattern,
                'priority': priority,
                'score': score
            })

        # ========== 4. 排序：优先级高的在前，同优先级按评分排序 ==========
        sorted_stocks = sorted(
            stock_sort_info,
            key=lambda x: (x['priority'], x['score']),
            reverse=True
        )

        # ========== 5. 提取排序后的股票代码 ==========
        result_stocks = [item['stock'] for item in sorted_stocks[:g.position_limit]]

        # ========== 6. 输出详细日志 ==========
#         log.info("=" * 60)
#         log.info("📊 股票排序详情（按模式优先级和评分）:")

        # 模式名称映射
        pattern_name_map = {
            "lb": "连板龙头",
            "yje": "一进二",
            "rzq": "弱转强",
            "dk": "首板低开",
            "fxsbdk": "反向首板低开",
            None: "无特定模式"
        }

        # 按模式分组统计
        pattern_count = {}
        for item in sorted_stocks:
            pattern = item['pattern']
            pattern_count[pattern] = pattern_count.get(pattern, 0) + 1

#         log.info(f"📈 各模式股票数量: {pattern_count}")
#         log.info("-" * 60)

        # 输出每只股票的详细信息
        for i, item in enumerate(sorted_stocks):
            pattern_name = pattern_name_map.get(item['pattern'], "未知模式")
            selected = "✓" if i < g.position_limit else " "

#             log.info(
#                 f"{selected} {i + 1:2d}. {item['stock']:12s} | "
#                 f"模式: {pattern_name:8s} | "
#                 f"优先级: {item['priority']} | "
#                 f"评分: {item['score']:2d}"
#             )

#         log.info("=" * 60)
#         log.info(f"✅ 最终选择 {len(result_stocks)} 只股票: {result_stocks}")

        # ========== 7. 输出选中股票的模式分布 ==========
        selected_pattern_count = {}
        for item in sorted_stocks[:g.position_limit]:
            pattern = item['pattern']
            pattern_name = pattern_name_map.get(pattern, "未知")
            selected_pattern_count[pattern_name] = selected_pattern_count.get(pattern_name, 0) + 1

#         log.info(f"📊 选中股票模式分布: {selected_pattern_count}")
#         log.info("=" * 60)

        return result_stocks

    except Exception as e:
#         log.error(f"❌ 股票排序失败: {str(e)}")
        import traceback
#         log.error(traceback.format_exc())
        # 返回原始列表
        return qualified_stocks[:g.position_limit]


def get_volume_data(stock_code, context=None):
    """
    获取股票的量能数据（最近两天成交量及量能比），带缓存功能

    Args:
        stock_code: 股票代码（如'603533.XSHG'）
        context: 上下文对象，用于获取当前日期，默认为None

    Returns:
        tuple: (last_volume, last_2_volume, trade_volume_ra)
            last_volume: 最近一个交易日的成交量
            last_2_volume: 倒数第二个交易日的成交量
            trade_volume_ra: 量能比（last_volume / last_2_volume），若数据不足或除零则为None
    """
    # 初始化返回值
    last_volume = None
    last_2_volume = None
    trade_volume_ra = None

    try:
        # 获取当前日期作为缓存键的一部分
        current_date = None
        if context:
            current_date = context.current_dt.strftime('%Y-%m-%d')
        else:
            # 如果没有提供context，尝试从全局变量获取当前日期
            try:
                from jqdata import get_trade_days
                import datetime
                current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            except:
                pass

        # 创建缓存键
        cache_key = f"{stock_code}_{current_date}" if current_date else None

        # 初始化全局缓存字典（如果不存在）
        if not hasattr(g, 'volume_data_cache'):
            g.volume_data_cache = {}

        # 如果有有效的缓存键且缓存中已存在数据，则直接返回缓存的结果
        if cache_key and cache_key in g.volume_data_cache:
            return g.volume_data_cache[cache_key]

        # 获取最近2个交易日的成交量数据（跳过停牌日）
        vol_hist = attribute_history(
            security=stock_code,
            count=2,
            unit='1d',
            fields=['volume'],
            skip_paused=True
        )

        # 校验数据有效性
        if vol_hist is not None and len(vol_hist) >= 2:
            # 提取成交量（倒数第二个交易日和最近一个交易日）
            last_2_volume = vol_hist['volume'].iloc[-2]  # 前一天成交量
            last_volume = vol_hist['volume'].iloc[-1]  # 当天成交量

            # 计算量能比（避免除零错误）
            if last_2_volume > 0:
                trade_volume_ra = round(last_volume / last_2_volume, 4)
            else:
#                 log.warning(f"[量能比计算警告] {stock_code} 前一天成交量为0，无法计算量能比")

                pass
        else:
#             log.warning(
#                 f"[量能数据不足] {stock_code} 有效交易日不足2天，获取到{len(vol_hist) if vol_hist is not None else 0}天数据")

        # 将结果存入缓存
            pass
        if cache_key:
            g.volume_data_cache[cache_key] = (last_volume, last_2_volume, trade_volume_ra)

    except Exception as e:
#         log.error(f"[量能获取失败] {stock_code} 错误原因: {str(e)}")

        pass
    return last_volume, last_2_volume, trade_volume_ra


def optimize_friday_trading_logic(context, qualified_stocks):
    """
    周五尾盘专用买入过滤。

    为什么要单独做周五逻辑：
    周五买入后要跨周末，消息面和不确定性更高，所以条件通常更苛刻。

    这里会额外检查：
    - 分数是否够；
    - 量能是否健康；
    - 市场环境是否太差；
    - 当前价是否站在 MA5 上方；
    - 开盘价是否明显高于现价（说明当天有回落，可能给出更低吸的位置）。
    """
    # 获取市场环境数据
    market_stats = g.trade_stats.get('market_stats', {})
    trend = market_stats.get('trend', '')
    volatility = market_stats.get('volatility', 0)
    volume_ratio = market_stats.get('volume_ratio', 0)
    filtered_stocks = []
    current_data = get_current_data()

#     log.info(f"股票g.score_cache 的评分内容 {g.score_cache}")
    for stock in qualified_stocks:
        try:
            # 检查评分缓存是否存在
            if stock not in g.score_cache:
#                 log.warning(f"股票 {stock} 的评分未在缓存中找到，跳过")
                continue

            # 从缓存中获取评分结果
            score_data = g.score_cache[stock]
            total_score = score_data.get('total_score', 0)

            # 获取股票当前数据
            stock_data = current_data[stock]
            current_price = stock_data.last_price
            open_price = stock_data.day_open  # 获取当日开盘价

            # 新增条件：开盘价比现价高2%以上（开盘价 > 现价 * 1.02）
            # 避免价格为0导致计算错误
            if current_price <= 0:
#                 log.warning(f"股票 {stock} 当前价格为0，跳过价格条件判断")
                continue
#             log.info(f"股票 {stock} 当前价格为{current_price}，open_price:{open_price}")
            open_vs_current = open_price > current_price * 1.02  # 开盘价较现价高2%以上

            # 修复bug：补全context参数，与buy主代码调用方式一致，避免函数执行失败
            last_volume, last_2_volume, trade_volume_ra = get_volume_data(stock, context)
            volume_energy = trade_volume_ra

            # 核心筛选条件
            conditions = []

            # 条件1: 评分要求（最低16分）
            conditions.append(total_score >= 16)

            # 条件2: 量能比要求（最低1.0）
            conditions.append(volume_energy >= 1.0)

            # 条件3: 市场环境过滤
            if trend in ['down', 'flat']:
                # 弱势市场提高要求
                conditions.append(total_score >= 18)
                conditions.append(volume_energy >= 1.2)

            # 条件4: 波动率过滤（避免过高波动）
            conditions.append(volatility <= 2.0)  # 最大2%波动

            # 条件5: 价格位置过滤（当前价在5日均线上方）
            ma5 = calculate_ma5(stock, context)
            conditions.append(current_price >= ma5 * 0.98)  # 允许2%偏差

            # 条件6: 买入原因优先级 - 适配修复后的get_buy_reason，增加健壮性兜底
            buy_reason = get_buy_reason(stock, context)
            # 增加日志：记录获取到的买入原因，便于周五筛选问题排查
            # 定义核心高优先级原因（连板龙头/弱转强），其余均按普通股票处理
            high_priority_reasons = ['连板龙头', '弱转强']
            if buy_reason in high_priority_reasons:
                # 龙头股放宽量能要求
                conditions.append(volume_energy >= 0.9)
            else:
                # 普通股票/未知原因/无特定模式，按原要求执行，避免分支缺失
                conditions.append(volume_energy >= 1.1)

            # 新增条件7: 开盘价较现价高2%以上
            conditions.append(open_vs_current)

            # 所有条件都满足
            if all(conditions):
                filtered_stocks.append(stock)

#                 log.info(f"✅ {stock} 符合尾盘买入条件 - "
#                          f"评分:{total_score} 量能:{volume_energy:.2f} "
#                          f"开盘/现价:{open_price:.2f}/{current_price:.2f}（高{((open_price / current_price) - 1) * 100:.2f}%） "
#                          f"原因:{buy_reason} 市场:{trend}")
        except Exception as e:
#             log.error(f"筛选 {stock} 出错：{str(e)}，跳过该股票")
            continue
    return filtered_stocks


def calculate_ma5(stock, context):
    """
    计算股票的5日均线（最近5个交易日收盘价的平均值）

    Args:
        stock: 股票代码（如'603533.XSHG'）
        context: 聚宽上下文对象

    Returns:
        float: 5日均线值（若数据不足则返回0）
    """
    try:
        # 获取最近5个交易日的收盘价数据（跳过停牌日）
        # 注意：使用'close'字段获取收盘价，单位为'1d'表示日线数据
        hist_data = attribute_history(
            security=stock,
            count=5,  # 获取5个交易日数据
            unit='1d',
            fields=['close'],  # 仅获取收盘价字段
            skip_paused=True  # 跳过停牌日
        )

        # 检查数据有效性（至少需要5个有效交易日数据）
        if hist_data is None or len(hist_data) < 5:
#             log.warning(
#                 f"股票 {stock} 有效交易日不足5天，当前可用数据量: {len(hist_data) if hist_data is not None else 0}")
            return 0.0

        # 计算5日收盘价平均值（即5日均线）
        ma5_value = hist_data['close'].mean()

        # 日志输出计算结果（调试用）
#         log.debug(f"股票 {stock} 5日均线计算完成: {ma5_value:.2f}（最近5日收盘价: {hist_data['close'].tolist()}）")

        return ma5_value

    except Exception as e:
#         log.error(f"计算股票 {stock} 5日均线失败: {str(e)}")
        return 0.0


def clear_score_cache(context):
    """
    清空评分缓存
    可以在每日开盘前调用
    """
    if hasattr(g, 'score_cache'):
        g.score_cache = {}
#         log.info("📭 评分缓存已清空")
    else:
        g.score_cache = {}


def calculate_buy_score_optimized(stock, context, money_flow_map):
    """
    计算单只股票的 6 因子总分。

    这是“单票评分器”。
    filter_stocks_by_score_optimized 会循环调用它，给每只候选股打分。

    计算流程：
    1. 取历史行情数据；
    2. 对齐资金流日期和收盘价日期；
    3. 分别计算 6 个因子分；
    4. 汇总成总分，并写入 g.score_cache。
    """
    try:
        # 1. 获取基础历史数据（含收盘价）
        required_fields = ['close', 'high', 'low', 'volume', 'high_limit']
        hist_data = attribute_history(
            stock,
            30,
            '1d',
            required_fields,
            skip_paused=True
        )

        # 2. 检查资金流数据
        fund_flow_list = money_flow_map.get(stock, [])
        has_money_data = len(fund_flow_list) >= 5

        # 3. 提取并对齐收盘价数据（与资金流日期匹配）
        # 这一步的意义：
        # 资金流数据和K线数据来源不同，先把日期对齐，再算资金因子才可靠。
        close_prices = []
        if not hist_data.empty and has_money_data:
            # 资金流日期列表（已排序）
            fund_dates = [pd.to_datetime(item['date']).date() for item in fund_flow_list]
            # 从历史数据中提取对应日期的收盘价
            for date in fund_dates:
                if date in hist_data.index.date:
                    # 找到对应日期的收盘价
                    close_price = hist_data.loc[hist_data.index.date == date, 'close'].values[0]
                    close_prices.append(close_price)
                else:
#                     log.warning(f"{stock} 资金流日期 {date} 无对应收盘价数据")
                    close_prices.append(0)  # 填充默认值

        # 4. 初始化 6 个因子分
        factor1_score = 0
        factor2_score = 0
        factor3_score = 0
        factor4_score = 0
        factor5_score = 0
        factor6_score = 0  # 主力资金因子

        # 5. 分项计算
        # 某一个因子报错时，不希望拖垮整只股票的总评分，所以这里尽量独立。
        factor1_score = calculate_limit_up_score_optimized(stock, context,
                                                           hist_data) if 'high_limit' in hist_data.columns else 0
        # TODO  一进二需要另外评估
        factor2_score = calculate_technical_score_optimized(stock, context, hist_data) if not hist_data.empty else 0
        factor3_score = calculate_volume_ma_score_optimized(stock, context,
                                                            hist_data) if 'volume' in hist_data.columns else 0
        factor4_score = calculate_mainline_score_optimized(stock, context)
        factor5_score = calculate_sentiment_score_optimized(stock, context)

        # 6. 主力资金因子
        # 这是比较“重”的一项，因为它会判断资金流规模、占市值比例、以及资金模式。
        factor6_score = calculate_main_force_flow_score(stock, fund_flow_list, close_prices)
        if not has_money_data and factor6_score > 0:
            factor6_score = int(factor6_score * 0.6)  # 数据不全时降权

        # 7. 总分计算
        total_score = sum([
            factor1_score, factor2_score, factor3_score,
            factor4_score, factor5_score, factor6_score
        ])

        # 8. 把结果先写入缓存。
        # 后面排序、打印买入原因、日志排查都会用到这份缓存。
        g.score_cache[stock] = {
            'total_score': total_score,
            'factors': {
                '涨停': factor1_score,
                '技术': factor2_score,
                '放量MA': factor3_score,
                '主线': factor4_score,
                '情绪': factor5_score,
                '主力资金': factor6_score
            }
        }

        # 9. 构建详细信息（修正资金数据来源）
        current_data = get_current_data()
        stock_info = current_data[stock] if stock in current_data else get_security_info(stock)
        # 提取资金流中的最近数据
        latest_fund_data = fund_flow_list[-1] if fund_flow_list else {}
        prev_fund_data = fund_flow_list[-2] if len(fund_flow_list) >= 2 else {}
        # 前三交易日资金流（取最近的3天）
        recent_3d_fund = fund_flow_list[-4:-1] if len(fund_flow_list) >= 4 else []

        details = {
            '股票名称': stock_info.name if hasattr(stock_info, 'name') else '未知',
            '当前价格': round(stock_info.last_price, 2) if hasattr(stock_info, 'last_price') else 0,
            '资金数据状态': '完整' if has_money_data else '缺失',
            '前一交易日主力净流入': round(prev_fund_data.get('net_amount_main', 0), 2) if prev_fund_data else 'N/A',
            '前三交易日平均净流入': round(
                sum(item.get('net_amount_main', 0) for item in recent_3d_fund) / len(recent_3d_fund),
                2) if recent_3d_fund else 'N/A',
            '评分时间': context.current_dt.strftime('%Y-%m-%d %H:%M:%S')
        }

        return {
            'total_score': total_score,
            'factor1_涨停': factor1_score,
            'factor2_技术': factor2_score,
            'factor3_放量MA': factor3_score,
            'factor4_主线': factor4_score,
            'factor5_情绪': factor5_score,
            'factor6_主力资金': factor6_score,
            'details': details,
            'cache_status': check_cache_status()
        }

    except Exception as e:
#         log.error(f"{stock} 买入评分计算失败：{str(e)}")
        return {
            'total_score': 0,
            'factor1_涨停': 0,
            'factor2_技术': 0,
            'factor3_放量MA': 0,
            'factor4_主线': 0,
            'factor5_情绪': 0,
            'factor6_主力资金': 0,
            'details': {'错误信息': str(e)},
            'cache_status': '计算失败'
        }


def calculate_main_force_flow_score(stock, fund_flow_list, close_prices):
    """
    计算“主力资金”因子分。

    这一项想回答的问题是：
    “最近几天主力资金到底有没有真正在做这只票，而且力度够不够强？”

    它不是只看单日净流入，而是同时看三件事：
    1. 资金占流通市值的比例；
    2. 绝对流入规模有多大；
    3. 近几天资金流的变化模式，是持续增强、突然爆发，还是弱转强。

    参数:
        stock: 股票代码
        fund_flow_list: 资金流字典列表（包含至少5天数据，需有'date'、'net_amount_main'字段）
        close_prices: 对应日期的收盘价列表（与fund_flow_list日期顺序一致）
    返回:
        资金流评分（整数分，分数越高表示资金信号越强）
    """
    try:
        # 1. 增强输入数据校验
        if not isinstance(fund_flow_list, list) or len(fund_flow_list) < 5:
#             log.warning(
#                 f"{stock} 资金流数据无效（非列表或不足5天），实际{len(fund_flow_list) if isinstance(fund_flow_list, list) else '非列表'}天，评0分")
            return 0

        required_fields = ['date', 'net_amount_main']
        for i, flow in enumerate(fund_flow_list):
            if not isinstance(flow, dict):
#                 log.warning(f"{stock} 资金流第{i + 1}条数据非字典格式，评0分")
                return 0
            missing_fields = [f for f in required_fields if f not in flow]
            if missing_fields:
#                 log.warning(f"{stock} 资金流第{i + 1}条缺失字段{missing_fields}，评0分")
                return 0

        if not isinstance(close_prices, list) or len(close_prices) != len(fund_flow_list):
#             log.warning(
#                 f"{stock} 收盘价数据无效（非列表或长度不匹配），资金流{len(fund_flow_list)}条，收盘价{len(close_prices) if isinstance(close_prices, list) else '非列表'}条")

        # 2. 数据预处理（去重+排序）
            pass
        hist_data = pd.DataFrame(fund_flow_list)
        hist_data['date'] = pd.to_datetime(hist_data['date'], errors='coerce')
        hist_data = hist_data.dropna(subset=['date'])
        if len(hist_data) < 5:
#             log.warning(f"{stock} 有效资金流数据不足5天（去重后{len(hist_data)}天），评0分")
            return 0

        hist_data = hist_data.drop_duplicates(subset=['date'], keep='last')
        hist_data = hist_data.sort_values('date').reset_index(drop=True)
        latest_idx = len(hist_data) - 1
        latest_date = hist_data['date'].iloc[-1]

        # 3. 提取关键数据
        # recent_5d_main: 最近5天主力净流入序列，是后续模式判断的核心输入。
        recent_5d_main = hist_data['net_amount_main'].tail(5).values  # 近5日主力净流入
        latest_main = recent_5d_main[-1]  # 前一交易日（最新）净流入
        ma5_flow = recent_5d_main.mean()  # 近5日MA5净流入

        # 计算前4天的资金流模式
        prev_4d_pattern = []
        for i in range(len(recent_5d_main) - 1):
            if recent_5d_main[i] < 0:
                prev_4d_pattern.append('-')
            elif recent_5d_main[i] > 0:
                prev_4d_pattern.append('+')
            else:
                prev_4d_pattern.append('0')

        pattern_str = ''.join(prev_4d_pattern)

        # 计算前4天平均值
        prev_4d_avg = np.mean(recent_5d_main[:-1]) if len(recent_5d_main) > 1 else 0

        # 计算爆发倍数
        explosion_multiple = latest_main / abs(prev_4d_avg) if abs(prev_4d_avg) > 0 else float('inf')

        # 4. 获取流通市值数据
        try:
            valuation_data = get_valuation(stock, end_date=latest_date.strftime('%Y-%m-%d'),
                                           count=1, fields=['circulating_market_cap'])
            if valuation_data.empty:
#                 log.warning(f"{stock} 无法获取流通市值数据，使用默认评分逻辑")
                circ_market_cap = None
                flow_to_market_ratio = None
            else:
                # 流通市值（亿元）
                circ_market_cap = valuation_data['circulating_market_cap'].iloc[0]
                # 资金流入占流通市值比例（百分比）
                flow_to_market_ratio = latest_main / (circ_market_cap * 10000) if circ_market_cap > 0 else 0
        except Exception as e:
#             log.warning(f"{stock} 获取流通市值失败: {str(e)}，使用默认评分逻辑")
            circ_market_cap = None
            flow_to_market_ratio = None

        # 5. 比例评分：
        # 把主力净流入和流通市值对比，判断这笔资金“相对公司盘子”算不算大。
        ratio_score = 0
        ratio_desc = "无比例数据"

        if flow_to_market_ratio is not None and circ_market_cap is not None:
            # 计算资金流入比例分数 - 极大拉开差距
            if flow_to_market_ratio >= 0.035:  # 3.5%以上
                ratio_score = 20  # 极高分数
                ratio_level = "极高比例"
            elif flow_to_market_ratio >= 0.03:  # 3%以上
                ratio_score = 16  # 超高分数
                ratio_level = "超高比例"
            elif flow_to_market_ratio >= 0.025:  # 2.5%以上
                ratio_score = 12  # 很高分数
                ratio_level = "很高比例"
            elif flow_to_market_ratio >= 0.02:  # 2%以上
                ratio_score = 8  # 高分数
                ratio_level = "高比例"
            elif flow_to_market_ratio >= 0.015:  # 1.5%以上
                ratio_score = 6
                ratio_level = "较高比例"
            elif flow_to_market_ratio >= 0.01:  # 1%以上
                ratio_score = 5
                ratio_level = "中高比例"
            elif flow_to_market_ratio >= 0.007:  # 0.7%以上
                ratio_score = 4
                ratio_level = "中等比例"
            elif flow_to_market_ratio >= 0.005:  # 0.5%以上
                ratio_score = 3
                ratio_level = "中低比例"
            elif flow_to_market_ratio >= 0.003:  # 0.3%以上
                ratio_score = 2
                ratio_level = "较低比例"
            elif flow_to_market_ratio > 0:  # 正值
                ratio_score = 1
                ratio_level = "低比例"
            else:  # 负值或零
                ratio_score = 0
                ratio_level = "无效比例"

            ratio_desc = f"{ratio_level} ({flow_to_market_ratio * 100:.4f}%)"

        # 6. 绝对规模评分：
        # 有些票盘子大，单看比例不够，还要看绝对金额够不够有存在感。
        absolute_score = 0
        if latest_main >= 50000:  # 5亿以上
            absolute_score = 10
            absolute_desc = "超大规模"
        elif latest_main >= 30000:  # 3亿以上
            absolute_score = 8
            absolute_desc = "大规模"
        elif latest_main >= 20000:  # 2亿以上
            absolute_score = 7
            absolute_desc = "中大规模"
        elif latest_main >= 10000:  # 1亿以上
            absolute_score = 6
            absolute_desc = "中等规模"
        elif latest_main >= 9000:  # 9000万以上 - 为603359特别调整
            absolute_score = 5.5
            absolute_desc = "中偏上规模"
        elif latest_main >= 7000:  # 7000万以上 - 为002313特别调整
            absolute_score = 4.5
            absolute_desc = "中偏小规模"
        elif latest_main >= 5000:  # 5000万以上
            absolute_score = 4
            absolute_desc = "中小规模"
        elif latest_main >= 2000:  # 2000万以上
            absolute_score = 3
            absolute_desc = "小规模"
        elif latest_main > 0:  # 正值
            absolute_score = 2
            absolute_desc = "微小规模"
        elif latest_main > -2000:  # 微负值
            absolute_score = 1
            absolute_desc = "微负规模"
        else:  # 负值
            absolute_score = 0
            absolute_desc = "负规模"

        # 7. 资金模式评分：
        # 关注的是“资金行为形态”，比如连续流出后突然转强，或者持续增强。
        pattern_score = 0

        # 处理爆发倍数的显示
        if explosion_multiple == float('inf'):
            explosion_multiple_str = '∞'
        else:
            explosion_multiple_str = f"{explosion_multiple:.2f}"

        pattern_desc = f"一般模式（模式: {pattern_str}，前4天平均: {prev_4d_avg:.2f}，爆发倍数: {explosion_multiple_str}）"

        # 连续4天净流入为负，最后一天大幅转正
        if pattern_str == '----' and latest_main > 0 and explosion_multiple > 5:
            pattern_score = 15
            pattern_desc = f"完美逆转（模式: {pattern_str}，爆发倍数: {explosion_multiple_str}）"
        # 连续3天净流入为负，最后两天转正且最后一天大于前一天
        elif pattern_str.endswith('-+') and latest_main > recent_5d_main[-2] > 0:
            pattern_score = 12
            pattern_desc = f"强势逆转（模式: {pattern_str}，最后两天比: {latest_main / recent_5d_main[-2]:.2f}）"
        # 连续4天净流入递增且最后一天为正
        elif all(recent_5d_main[i] < recent_5d_main[i + 1] for i in range(len(recent_5d_main) - 1)) and latest_main > 0:
            pattern_score = 8  # 保持8分
            pattern_desc = f"持续增强（模式: 递增，最后一天: {latest_main:.2f}）"
        # 最后一天资金净流入为正且大于前4天平均值的3倍
        elif latest_main > 0 and prev_4d_avg > 0 and latest_main > prev_4d_avg * 3:
            pattern_score = 6
            pattern_desc = f"突然爆发（爆发倍数: {latest_main / prev_4d_avg:.2f}）"
        # 最后一天资金净流入为正且大于前4天平均值
        elif latest_main > 0 and latest_main > prev_4d_avg:
            pattern_score = 4
            pattern_desc = f"温和增强（比前均值: {latest_main / prev_4d_avg:.2f}倍）"

        # 8. 三部分加权合成最终分数
        ratio_weight = 0.40  # 资金流入比例权重提高到40%
        absolute_weight = 0.20  # 绝对资金规模权重提高到20%
        pattern_weight = 0.40  # 资金模式权重降低到40%

        # 9. 总分计算与日志
        weighted_score = (ratio_score * ratio_weight +
                          absolute_score * absolute_weight +
                          pattern_score * pattern_weight)

        # 确保总分不超过10分
        # weighted_score = min(weighted_score, 10)

        # 四舍五入到整数
        total = round(weighted_score)

        # 近5日是否逐步递增（仅供参考）
        is_increasing = all(recent_5d_main[i] < recent_5d_main[i + 1] for i in range(len(recent_5d_main) - 1))

        # 流通市值相关日志
        market_cap_info = ""
        if circ_market_cap is not None:
            market_cap_info = f"流通市值: {circ_market_cap:.2f}亿元，资金流入占比: {flow_to_market_ratio * 100:.4f}%，"

#         log.info(
#             f"{stock} 资金流评分明细：\n"
#             f"  近5日净流入数据: {[round(x, 2) for x in recent_5d_main]}\n"
#             f"  近5日MA5净流入: {ma5_flow:.2f}，前一交易日净流入: {latest_main:.2f}\n"
#             f"  {market_cap_info}\n"
#             f"  资金流入比例评分: {ratio_desc} → {ratio_score}分 (权重{ratio_weight * 100}%)\n"
#             f"  绝对资金规模评分: {absolute_desc}（{latest_main:.2f}万） → {absolute_score}分 (权重{absolute_weight * 100}%)\n"
#             f"  资金模式评分: {pattern_desc} → {pattern_score}分 (权重{pattern_weight * 100}%)\n"
#             f"  近5日是否逐步递增: {'是' if is_increasing else '否'} (仅供参考)\n"
#             f"  加权总分: {weighted_score:.2f} → 最终总分: {total}"
#         )
        return total

    except KeyError as e:
#         log.error(f"{stock} 资金流字段缺失: {str(e)}")
        return 0
    except IndexError as e:
#         log.error(f"{stock} 数据索引错误: {str(e)}")
        return 0
    except Exception as e:
#         log.error(f"{stock} 资金流评分计算失败: {str(e)}")
        import traceback
#         log.error(traceback.format_exc())
        return 0


# ============================================================================
# 2. 评分计算相关函数（修复版）
# ============================================================================

def calculate_limit_up_score_optimized(stock, context, hist_data=None):
    """
    计算“涨停强度”因子分。

    核心思想：
    昨天如果是强势涨停，或者非常接近涨停，同时成交量也配合，
    说明这只票有较强的短线攻击性。

    参数:
    pass
    stock: 股票代码
    context: 聚宽上下文
    hist_data: 历史数据（可选，如果不提供则内部获取）

    返回:
    pass
    int: 涨停评分 (0-5分)
    """
    try:
        # 如果没有提供历史数据，则获取
        if hist_data is None:
            hist_data = attribute_history(stock, 10, '1d',
                                          ['close', 'high', 'low', 'volume', 'high_limit'],
                                          skip_paused=True)

        # 至少需要有1条数据（昨日数据）
        if hist_data.empty or len(hist_data) < 1:
            return 0

        score = 0

        # 1. 价格强度：看昨天是否真正封涨停，或者至少非常接近涨停
        # 关键修复：使用iloc[-1]获取最后一条数据（实际昨日数据）
        yesterday_close = hist_data['close'].iloc[-1]
        yesterday_high_limit = hist_data['high_limit'].iloc[-1]
        yesterday_high = hist_data['high'].iloc[-1]  # 昨日最高价

        # 数据有效性校验
        if pd.isna(yesterday_close) or pd.isna(yesterday_high_limit) or yesterday_high_limit <= 0:
            return 0

        # 计算相对误差（更适应不同股价）
        price_diff = abs(yesterday_close - yesterday_high_limit)
        relative_diff = price_diff / yesterday_high_limit  # 相对误差比例

        # 涨停判断（相对误差≤0.1%）
        if relative_diff <= 0.001:
            score += 3
            log_msg = f"{stock} 昨日涨停（收盘价: {yesterday_close:.2f}, 涨停价: {yesterday_high_limit:.2f}, 相对误差: {relative_diff:.4%}），+3分"
#             log.debug(log_msg)

        # 精细化接近涨停判断
        else:
            # 情况1：收盘价≥95%涨停价且最高价接近涨停（冲板未封死）
            if (yesterday_close >= yesterday_high_limit * 0.95) and (yesterday_high >= yesterday_high_limit * 0.995):
                score += 2
                log_msg = f"{stock} 昨日冲板未封死（收盘价: {yesterday_close:.2f}, 最高价: {yesterday_high:.2f}），+2分"
#                 log.debug(log_msg)

            # 情况2：仅收盘价≥95%涨停价（未冲板）
            elif yesterday_close >= yesterday_high_limit * 0.95:
                score += 1
                log_msg = f"{stock} 昨日收盘价接近涨停（{yesterday_close:.2f}/{yesterday_high_limit:.2f}），+1分"
#                 log.debug(log_msg)

        # 2. 量能质量：同样是涨停，放量涨停通常比缩量涨停更有辨识度
        yesterday_volume = hist_data['volume'].iloc[-1]  # 修复：昨日成交量取最后一条
        # 计算昨日之前的平均成交量（排除昨日）
        prev_volumes = hist_data['volume'].iloc[:-1]  # 取截止到昨日之前的所有成交量
        if len(prev_volumes) < 1:
            return min(score, 5)

        avg_volume = prev_volumes.mean()

        # 放量判断
        if yesterday_volume > avg_volume * 1.5:
            score += 2
            log_msg = f"{stock} 放量涨停（昨日成交量: {yesterday_volume}, 平均成交量: {avg_volume:.2f}），+2分"
#             log.debug(log_msg)
        elif yesterday_volume > avg_volume * 1.2:
            score += 1
            log_msg = f"{stock} 适度放量涨停（昨日成交量: {yesterday_volume}, 平均成交量: {avg_volume:.2f}），+1分"
#             log.debug(log_msg)

        final_score = min(score, 5)

        return final_score

    except Exception as e:
        error_msg = f"计算涨停评分失败 {stock}: {str(e)}"
#         log.error(error_msg)
        return 0


def calculate_technical_score_optimized(stock, context, hist_data=None):
    """
    计算“技术形态”因子分。

    这部分主要看三件事：
    1. 近10日涨停活跃度；
    2. 均线结构是否偏强；
    3. RSI 和价格区间位置是否健康。

    总分范围：0-10分。
    """
    try:
        # 1. 获取历史数据（补充涨停价字段用于判断涨停）
        if hist_data is None:
            hist_data = attribute_history(stock, 30, '1d',
                                          ['close', 'high', 'low', 'volume', 'high_limit'],  # 新增high_limit字段
                                          skip_paused=True)

        # 数据有效性校验（至少需要10个交易日数据）
        if hist_data.empty or len(hist_data) < 10:
#             log.warning(f"{stock} 历史数据不足10个交易日，技术评分为0")
            return 0

        score = 0
        close_prices = hist_data['close']

        # 2. 近10日涨停数
        # 这里更像是在看“股性是否活跃”，而不是看单日走势。
        # 取最近10个交易日数据（含当日）
        recent_10d = hist_data.tail(10)
        # 过滤无效数据（涨停价为0或NaN的情况）
        valid_days = recent_10d[(recent_10d['high_limit'] > 0) &
                                (recent_10d['high_limit'].notna()) &
                                (recent_10d['close'].notna())]

        # 计算涨停天数（收盘价等于涨停价视为涨停）
        limit_up_count = sum(valid_days['close'] == valid_days['high_limit'])

        # 根据涨停数计分
        if limit_up_count == 0:
            limit_up_score = 0
        elif 1 <= limit_up_count <= 3:
            limit_up_score = 2
        elif 4 <= limit_up_count <= 5:
            limit_up_score = 3
        else:  # >5个涨停
            limit_up_score = 5

#         log.debug(f"{stock} 近10日涨停数: {limit_up_count}，活跃度得分: {limit_up_score}")
        score += limit_up_score  # 加入总评分

        # 3. 均线结构
        # 多头排列通常表示趋势更顺。
        if len(close_prices) >= 20:
            ma5 = close_prices.rolling(window=5).mean().iloc[-1]
            ma10 = close_prices.rolling(window=10).mean().iloc[-1]
            ma20 = close_prices.rolling(window=20).mean().iloc[-1]
            current_price = close_prices.iloc[-1]

            # 多头排列判断
            if current_price > ma5 > ma10 > ma20:
                score += 2
#                 log.debug(f"{stock} 均线多头排列（强），+2分")
            elif current_price > ma5 > ma10:
                score += 1
#                 log.debug(f"{stock} 均线多头排列（弱），+1分")
            else:
#                 log.debug(f"{stock} 非多头排列，均线得0分")

        # 4. RSI
        # 不是越高越好，太高太低都可能意味着位置不舒服。
                pass
        if len(close_prices) >= 14:
            rsi = calculate_rsi(close_prices, 14)
            rsi_score = 0
            if 30 <= rsi <= 70:
                rsi_score += 1
            if 40 <= rsi <= 60:
                rsi_score += 1
            score += rsi_score
#             log.debug(f"{stock} RSI({rsi:.1f})得分: {rsi_score}")

        # 5. 价格区间位置
        # 当前价处于20日区间上半部，说明相对更强。
        if len(hist_data) >= 20:
            high_20 = hist_data['high'].tail(20).max()
            low_20 = hist_data['low'].tail(20).min()
            current_price = close_prices.iloc[-1]

            # 避免高低点相同导致除零错误
            if high_20 > low_20 and current_price > (high_20 + low_20) / 2:
                score += 1
#                 log.debug(f"{stock} 价格在20日区间上半部分，+1分")
            else:
#                 log.debug(f"{stock} 价格在20日区间下半部分，位置得0分")

        # 总分上限调整为10分（活跃度5分+原有技术分5分）
                pass
        final_score = min(score, 10)
#         log.debug(f"{stock} 最终技术评分: {final_score}")
        return final_score

    except Exception as e:
#         log.error(f"计算技术评分失败 {stock}: {str(e)}")
        return 0


def calculate_rsi(prices, period=14):
    """
    计算RSI指标
    """
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except:
        return 50  # 默认返回中性值


def calculate_volume_ma_score_optimized(stock, context, hist_data=None):
    """
    计算“量价配合”因子分。

    关注两类信号：
    1. 价格是否站上/走强于短均线；
    2. 最近成交量是否明显高于过去平均水平。
    """
    try:
        # 如果没有提供历史数据，则获取
        if hist_data is None:
            hist_data = attribute_history(stock, 30, '1d',
                                          ['close', 'volume'],
                                          skip_paused=True)

        if hist_data.empty or len(hist_data) < 10:
            return 0

        score = 0
        close_prices = hist_data['close']
        volume_data = hist_data['volume']

        # 1. 均线突破信号
        if len(close_prices) >= 10:
            ma5_current = close_prices.rolling(window=5).mean().iloc[-1]
            ma5_yesterday = close_prices.rolling(window=5).mean().iloc[-2]
            ma10_current = close_prices.rolling(window=10).mean().iloc[-1]
            current_price = close_prices.iloc[-1]

            # 价格突破MA5
            if current_price > ma5_current > ma5_yesterday:
                score += 2
            elif current_price > ma5_current:
                score += 1

            # MA5上穿MA10
            if ma5_current > ma10_current and ma5_yesterday <= ma10_current:
                score += 1

        # 2. 放量确认
        # 有价还要有量，量能配合越明显，分越高。
        if len(volume_data) >= 5:
            recent_volume = volume_data.tail(3).mean()
            historical_volume = volume_data.head(-3).mean()

            if recent_volume > historical_volume * 1.5:
                score += 2
            elif recent_volume > historical_volume * 1.2:
                score += 1

        return min(score, 5)  # 最高5分

    except Exception as e:
#         log.error(f"计算放量MA评分失败 {stock}: {str(e)}")
        return 0


# 4. 盘后统计函数
# 针对"不在价格数据中"错误的优化代码（主要修改盘后统计相关逻辑）
def record_closing_stats(context):
    """盘后数据统计函数（优化版）"""
    try:
#         log.info("\n" + "=" * 60)
#         log.info(f"==== 盘后数据统计 [{context.current_dt.strftime('%Y-%m-%d %H:%M')}] ====")

        # 1. 账户核心信息统计
        portfolio = context.portfolio
        account_stats = {
            # "总权益": round(portfolio.total_value, 2),
            # "可用资金": round(portfolio.available_cash, 2),
            "持仓总价值": round(portfolio.positions_value, 2),
            # "累计出入金": round(portfolio.inout_cash, 2),
            # "累计收益": f"{portfolio.returns:.2%}",
            # "初始资金": round(portfolio.starting_cash, 2),
            # "可取资金": round(portfolio.transferable_cash, 2),
            # "锁住资金": round(portfolio.locked_cash, 2)
        }

        # log.info("\n----- 账户核心信息 -----")
        for key, value in account_stats.items():
            log.info(f"{key}: {value}")

        # 2. 持仓标的统计（替代valid_stocks逻辑）
            pass
        valid_stocks = []
        long_positions = portfolio.long_positions
        short_positions = portfolio.short_positions

        # 多单持仓统计
        long_count = len(long_positions)
        valid_stocks.extend([pos.security for pos in long_positions.values() if pos.total_amount > 0])

        # 空单持仓统计
        short_count = len(short_positions)
        valid_stocks.extend([pos.security for pos in short_positions.values() if pos.total_amount > 0])

        # 去重处理
        valid_stocks = list(set(valid_stocks))
#         log.info(f"\n----- 持仓概览 -----")
#         log.info(f"有效持仓标的数量: {len(valid_stocks)}")
#         log.info(f"多单持仓数量: {long_count}")
#         log.info(f"空单持仓数量: {short_count}")

        # 3. 多单持仓详情
#         log.info("\n----- 多单持仓详情 -----")
        if long_positions:
            for pos in long_positions.values():
                if pos.total_amount <= 0:
                    continue
                pos_info = (
                    f"标的: {pos.security} | "
                    f"总仓位: {pos.total_amount} | "
                    f"可平仓数量: {pos.closeable_amount} | "
                    f"最新价: {pos.price:.2f} | "
                    f"持仓价值: {pos.value:.2f} | "
                    f"累计成本: {pos.acc_avg_cost:.2f} | "
                    f"建仓时间: {pos.init_time.strftime('%Y-%m-%d')}"
                )
#                 log.info(pos_info)
        else:
#             log.info("无多单持仓")

        # 4. 空单持仓详情
#         log.info("\n----- 空单持仓详情 -----")
            pass
        if short_positions:
            for pos in short_positions.values():
                if pos.total_amount <= 0:
                    continue
                pos_info = (
                    f"标的: {pos.security} | "
                    f"总仓位: {pos.total_amount} | "
                    f"可平仓数量: {pos.closeable_amount} | "
                    f"最新价: {pos.price:.2f} | "
                    f"持仓价值: {pos.value:.2f} | "
                    f"累计成本: {pos.acc_avg_cost:.2f} | "
                    f"建仓时间: {pos.init_time.strftime('%Y-%m-%d')}"
                )
#                 log.info(pos_info)
        else:
#             log.info("无空单持仓")

        # 5. 交易统计更新
            pass
        if hasattr(g, 'trade_stats'):
            # 记录当日收益
            current_return = (portfolio.total_value / portfolio.starting_cash) - 1
            g.trade_stats['daily_returns'].append({
                "date": context.current_dt.date(),
                "return": current_return
            })

            # 记录持仓统计
            g.trade_stats['position_stats'][context.current_dt.date()] = {
                "long_count": long_count,
                "short_count": short_count,
                "total_value": portfolio.total_value
            }

#             log.info("\n----- 交易统计更新 -----")
#             log.info(f"当日收益: {current_return:.2%}")
#             log.info(f"累计交易日: {len(g.trade_stats['daily_returns'])}")

        # 6. 热门概念缓存状态
#         log.info(f"\n----- 系统状态 -----")
#         log.info(f"热门概念缓存状态: {check_cache_status()}")
#         log.info("=" * 60 + "\n")

    except Exception as e:
#         log.error(f"{context.current_dt.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - 盘后数据统计失败: {str(e)}")
        import traceback
#         log.error(
#             f"{context.current_dt.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - Traceback (most recent call last):\n{traceback.format_exc()}")


# 5. 辅助函数
def get_buy_reason(stock, context):
    """
    返回股票属于哪一种买入模式。

    这个函数不负责打分，只负责给结果贴标签。
    主要用途有两个：
    1. 买入日志里告诉你“为什么买它”；
    2. 某些场景下，按模式区分不同阈值。

    参数:
    pass
    stock: 股票代码
    context: 上下文对象
    返回:
    pass
    买入原因描述
    """
    try:
        # 按buy代码中的模式优先级排序判断，同时检查全局筛选后列表是否存在
        # 优先级：连板龙头 → 弱转强 → 一进二 → 首板低开 → 反向首板低开
        if hasattr(g, 'lblt_stocks') and stock in g.lblt_stocks:
            return "连板龙头"
        elif hasattr(g, 'rzq_stocks') and stock in g.rzq_stocks:
            return "弱转强"
        elif hasattr(g, 'gk_stocks') and stock in g.gk_stocks:
            return "一进二"
        elif hasattr(g, 'dk_stocks') and stock in g.dk_stocks:
            return "首板低开"
        elif hasattr(g, 'fxsbdk_stocks') and stock in g.fxsbdk_stocks:
            return "反向首板低开"
        else:
            return "无特定模式"
    except Exception as e:
#         log.error(f"获取 {stock} 买入原因失败: {str(e)}")
        return "未知原因"


# 处理日期相关函数
def transform_date(date, date_type):
    """
    在字符串、datetime、date 三种日期格式之间转换。

    这个函数本身不参与交易逻辑，只是为了让其它函数少写重复的日期转换代码。
    """
    if type(date) == str:
        str_date = date
        dt_date = dt.datetime.strptime(date, '%Y-%m-%d')
        d_date = dt_date.date()
    elif type(date) == dt.datetime:
        str_date = date.strftime('%Y-%m-%d')
        dt_date = date
        d_date = dt_date.date()
    elif type(date) == dt.date:
        str_date = date.strftime('%Y-%m-%d')
        dt_date = dt.datetime.strptime(str_date, '%Y-%m-%d')
        d_date = date
    dct = {'str': str_date, 'dt': dt_date, 'd': d_date}
    return dct[date_type]


# 过滤函数
def filter_new_stock(initial_list, date, days=50):
    """过滤上市时间太短的新股，避免样本太新、波动过大。"""
    d_date = transform_date(date, 'd')
    return [stock for stock in initial_list if d_date - get_security_info(stock).start_date > dt.timedelta(days=days)]


def filter_st_paused_stock(initial_list):
    """过滤 ST、停牌、退市整理类股票。"""
    current_data = get_current_data()
    # 使用列表推导式结合any()函数，筛选出符合条件的股票
    return [stock for stock in initial_list
            if not any([
            current_data[stock].is_st,  # 排除ST股
            current_data[stock].paused,  # 排除停牌股
            '退' in current_data[stock].name  # 排除名称中含'退'字的股票，避免退市股
        ])]


def filter_kcbj_stock(initial_list):
    """只保留主板/创业板常见代码，排除科创板、北交所等不想参与的市场。"""
    return [stock for stock in initial_list if stock[:2] in ('60', '00', '30')]


# 每日初始股票池
def prepare_stock_list(date):
    """
    生成每日的基础股票池。

    这是所有后续选股的最上游入口。
    顺序上就是：
    全市场股票 -> 去掉不想做的市场 -> 去掉新股 -> 去掉 ST/停牌/退市风险股。
    """
    initial_list = get_all_securities('stock', date).index.tolist()
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_new_stock(initial_list, date)
    initial_list = filter_st_paused_stock(initial_list)
    return initial_list


def rise_low_volume(s, context):  # 上涨时，未放量 rising on low volume
    """
    检查“左压”附近是否缩量。

    通俗说：
    如果股价正在靠近前高，但成交量没有异常放大，
    说明抛压可能没那么重，向上突破时更干净。

    这是一个偏辅助的形态判断函数。
    """
    try:
        hist = attribute_history(s, 106, '1d', fields=['high', 'volume'], skip_paused=True, df=False)
        if hist is None or len(hist['high']) < 102:
#             log.info(f"左压检查 {s}: 历史数据不足102天，跳过检查，返回False")
            return False

        high_prices = hist['high'][:102]
        prev_high = high_prices[-1]
        # 查找前一个高点
        zyts_0 = next((i - 1 for i, high in enumerate(high_prices[-3::-1], 2) if high >= prev_high), 100)
        zyts = zyts_0 + 5

        # 确保索引有效
        if zyts > len(hist['volume']):
#             log.info(f"左压检查 {s}: zyts={zyts} 超出数据长度，返回False")
            return False

        last_volume = hist['volume'][-1]
        max_volume_past = max(hist['volume'][-zyts:-1]) if zyts > 1 else 0
        condition = last_volume <= max_volume_past * 0.9 if max_volume_past > 0 else False
        
        if condition == False:
#             log.info(f"左压检查 {s}: prev_high={prev_high:.2f}, zyts={zyts}, 最近成交量={last_volume}, "
#                  f"过去{zyts - 1}天最大成交量={max_volume_past}, 条件(最近<=过去最大*0.9)={condition}, 返回{condition}")
            pass
        return condition
    except Exception as e:
#         log.error(f"左压检查 {s} 出错: {e}")
        return False


# 筛选出某一日涨停的股票
def get_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close', 'high_limit'], count=1, panel=False,
                   fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    df = df[df['close'] == df['high_limit']]
    hl_list = list(df.code)
    return hl_list


# 筛选曾涨停
def get_ever_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['high', 'high_limit'], count=1, panel=False,
                   fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    df = df[df['high'] == df['high_limit']]
    hl_list = list(df.code)
    return hl_list


# 筛选曾涨停
def get_ever_hl_stock2(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close', 'high', 'high_limit'], count=1,
                   panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    cd1 = df['high'] == df['high_limit']
    cd2 = df['close'] != df['high_limit']
    df = df[cd1 & cd2]
    hl_list = list(df.code)
    return hl_list


# 计算涨停数
def get_hl_count_df(hl_list, date, watch_days):
    # 获取watch_days的数据
    df = get_price(hl_list, end_date=date, frequency='daily', fields=['close', 'high_limit', 'low'], count=watch_days,
                   panel=False, fill_paused=False, skip_paused=False)
    df.index = df.code
    # 计算涨停与一字涨停数，一字涨停定义为最低价等于涨停价
    hl_count_list = []
    extreme_hl_count_list = []
    for stock in hl_list:
        df_sub = df.loc[stock]
        hl_days = df_sub[df_sub.close == df_sub.high_limit].high_limit.count()
        extreme_hl_days = df_sub[df_sub.low == df_sub.high_limit].high_limit.count()
        hl_count_list.append(hl_days)
        extreme_hl_count_list.append(extreme_hl_days)
    # 创建df记录
    df = pd.DataFrame(index=hl_list, data={'count': hl_count_list, 'extreme_count': extreme_hl_count_list})
    return df


# 计算连板数
def get_continue_count_df(hl_list, date, watch_days):
    df = pd.DataFrame()
    for d in range(2, watch_days + 1):
        HLC = get_hl_count_df(hl_list, date, d)
        CHLC = HLC[HLC['count'] == d]
        df = df.append(CHLC)
    stock_list = list(set(df.index))
    ccd = pd.DataFrame()
    for s in stock_list:
        tmp = df.loc[[s]]
        if len(tmp) > 1:
            M = tmp['count'].max()
            tmp = tmp[tmp['count'] == M]
        ccd = ccd.append(tmp)
    if len(ccd) != 0:
        ccd = ccd.sort_values(by='count', ascending=False)
    return ccd


def record_sell_trade(context, stock, reason, details, current_data, date):
    """
    记录一笔卖出交易。

    它本身不负责卖出决策，只负责“记账”：
    - 记下卖出原因；
    - 计算这笔卖出的盈亏；
    - 写入 g.today_trades，供盘后汇总使用。
    """
    try:
        position = context.portfolio.positions[stock]
        avg_cost = position.avg_cost
        current_price = current_data[stock].last_price
        profit_pct = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0

        # 构建交易记录
        trade_record = {
            'stock': stock,
            'action': '卖出',
            'price': current_price,
            'reason': reason,
            'details': details,
            'profit_pct': profit_pct,
            'date': context.current_dt.date()  # 记录交易日期
        }

        # 记录到今日交易
        if not hasattr(g, 'today_trades'):
            g.today_trades = []
        g.today_trades.append(trade_record)
        log.info(f"卖出执行: {stock}, 原因={reason}, 价格={current_price:.2f}, 盈亏={profit_pct:.2%}")

        # 关键：更新上一笔交易信息
        g.last_trade_info = {
            'date': context.current_dt.date(),
            'profit_pct': profit_pct
        }
#         log.info(f"更新上一笔交易信息: {g.last_trade_info}")

    except Exception as e:
#         log.error(f"记录卖出交易失败: {str(e)}")


# ================== 安全历史数据封装 ==================
        pass
ALLOWED_FIELDS = {
    'open', 'close', 'high', 'low', 'volume', 'money', 'avg',
    'high_limit', 'low_limit', 'pre_close', 'paused', 'factor', 'open_interest'
}


def get_trading_time_status(context):
    """
    判断当前时间属于哪个时段。

    返回三个布尔值：
    - is_morning: 是否处于上午大时段
    - is_afternoon: 是否处于下午大时段
    - is_trading_time: 是否处于真正连续竞价时间

    这个函数很基础，但很重要，因为很多买卖规则只在某个时间段生效。
    Args:
        context: 聚宽上下文对象
    Returns:
        tuple: (is_morning, is_afternoon, is_trading_time)
    """
    import datetime as dt

    # 获取当前时间（确保是北京时间）
    current_dt = context.current_dt
    current_time = current_dt.time()  # 时间对象（时:分:秒）
    current_datetime_str = current_dt.strftime("%Y-%m-%d %H:%M:%S")  # 完整时间字符串，用于日志

    # 定义A股交易时间段（北京时间）
    morning_start = dt.time(7, 30)  # 上午开始时间
    morning_end = dt.time(11, 30)  # 上午结束时间
    afternoon_start = dt.time(13, 0)  # 下午开始时间
    afternoon_end = dt.time(19, 0)  # 下午结束时间
    trade_morning_start = dt.time(9, 30)  # 交易上午开始时间
    trade_morning_end = dt.time(11, 30)  # 交易上午结束时间
    trade_afternoon_start = dt.time(13, 0)  # 交易下午开始时间
    trade_afternoon_end = dt.time(15, 0)  # 交易下午结束时间

    # 明确判断逻辑（拆分链式比较，增强可读性）
    is_morning = (current_time >= morning_start) and (current_time <= morning_end)
    is_afternoon = (current_time >= afternoon_start) and (current_time <= afternoon_end)
    is_trade_morning = (current_time >= trade_morning_start) and (current_time <= trade_morning_end)
    is_trade_afternoon = (current_time >= trade_afternoon_start) and (current_time <= trade_afternoon_end)
    is_trading_time = is_trade_morning or is_trade_afternoon

    return is_morning, is_afternoon, is_trading_time


def get_5min_volume_ratio(stock, context, period=5):
    """
    计算 5 分钟量比。

    公式很简单：
    当前这一根 5 分钟K线成交量 / 过去 period 根 5 分钟K线平均成交量

    作用：
    用来判断这只股票此刻是不是突然放量。

    参数:
        stock: 股票代码
        context: 聚宽上下文
        period: 用于计算均量的K线数量（默认5）
    返回:
        ratio: 当前量 / 平均量（若数据不足或当前量太小返回None）
    """
    try:
        # 获取 period+1 根5分钟K线（包含当前未完结的K线）
        # 使用 datetime 对象作为 end_dt
        bars = get_bars(
            security=stock,
            count=period + 1,
            unit='5m',
            fields=['date', 'volume'],  # 仅需要日期和成交量
            include_now=True,
            end_dt=context.current_dt,  # 直接使用 datetime 对象
            df=True
        )
        if bars is None or len(bars) < period + 1:
#             log.debug(f"{stock} 5分钟K线数据不足{period + 1}根，实际{len(bars) if bars is not None else 0}")
            return None

        # 最新一根（当前K线）的成交量
        current_vol = bars['volume'].iloc[-1]
        # 如果当前成交量极小（如刚开盘），视为未放量
        if current_vol <= 0:
            return 0.0

        # 前period根的平均成交量（排除当前根）
        prev_vols = bars['volume'].iloc[-period - 1:-1]
        if prev_vols.mean() <= 0:
            return None

        ratio = current_vol / prev_vols.mean()
        return ratio
    except Exception as e:
#         log.error(f"获取{stock}5分钟量比失败: {e}")
        return None


def get_1min_volume_ratio(stock, context, period=5):
    """
    计算 1 分钟量比。

    和 get_5min_volume_ratio 思路一样，只是换成 1 分钟粒度。
    主要用于 09:32 这种非常早的时点，因为这时 5 分钟K线还不够灵敏。
    """
    try:
        bars = get_bars(
            security=stock,
            count=period + 1,
            unit='1m',
            fields=['date', 'volume'],
            include_now=True,
            end_dt=context.current_dt,
            df=True
        )
        if bars is None or len(bars) < period + 1:
            return None
        current_vol = bars['volume'].iloc[-1]
        if current_vol <= 0:
            return 0.0
        prev_vols = bars['volume'].iloc[-period-1:-1]
        if prev_vols.mean() <= 0:
            return None
        return current_vol / prev_vols.mean()
    except Exception as e:
#         log.error(f"获取{stock}1分钟量比失败: {e}")
        return None


# ================== 主卖出逻辑 ==================
def sell2(context):
    """
    主卖出函数。

    这是整份脚本里最复杂的函数之一，因为它把多套卖出逻辑放在一起：
    1. 上午偏风控，处理止损、走弱、放量异常；
    2. 下午偏止盈，尤其关注 14:50 左右的尾盘量能；
    3. 全程记录卖出原因，方便复盘。

    第一次阅读时，建议先抓主线：
    “先按上午/下午分支，再看每个时间段有哪些触发条件”。
    """
    # 初始化交易记录
    if not hasattr(g, 'today_trades'):
        g.today_trades = []

    # 获取当前市场数据
    current_data = get_current_data()
    date = transform_date(context.previous_date, 'str')
    current_dt = context.current_dt  # 完整datetime，用于筛选/取数
    current_time = current_dt.time()
    today_date = current_dt.date()  # 当日日期，用于筛选5分钟K线
    today = context.current_dt
    # 判断当前时间段
    is_morning, is_afternoon, is_trading_time = get_trading_time_status(context)
    # 14:50尾盘专属时段（核心放量卖出时段）
    is_profit_taking_time = (current_time >= dt.time(14, 50) and
                             current_time < dt.time(15, 0))

    # 判断当前时间段
    is_morning, is_afternoon, is_trading_time = get_trading_time_status(context)

    # 性能优化：
    # 先批量拉取持仓股历史数据，避免在循环里一只只查，减少 API 调用次数。
    slist = [s for s in context.portfolio.positions if context.portfolio.positions[s].closeable_amount > 0]
    hist_map = {}
    if slist:
        hist_df = get_price(slist, end_date=date, count=4,
                            fields=['close', 'high', 'volume'], panel=False, skip_paused=True)
        if hist_df is not None and not hist_df.empty:
            hist_map = {code: df.sort_values('time').reset_index(drop=True)
                        for code, df in hist_df.groupby('code')}

    # 遍历持仓股票
    for stock in list(context.portfolio.positions):
        try:
            position = context.portfolio.positions[stock]

            # 跳过不可平仓的股票
            if position.closeable_amount == 0:
                continue

            # 获取股票当前信息
            current_price = current_data[stock].last_price
            avg_cost = position.avg_cost
            high_limit = current_data[stock].high_limit

            # 跳过停牌股票
            if current_data[stock].paused:
#                 log.info(f"{stock} 今日停牌，跳过卖出检查")
                continue

            # 上午时间段卖出策略
            if is_morning:
                # 1. 月度一号时间止损策略
                try:
                    hist = history(10, '1d', 'open', [stock], df=False)
                    if len(hist.get(stock, [])) == 10:
                        start_price = hist[stock][0]
                        end_price = hist[stock][-1]

                        # 10日涨幅大于80%且月初未涨停
                        if end_price / start_price > 1.8 and today.day == 1 and (high_limit > current_price):
                            details = {
                                '10日涨幅': f"{(end_price / start_price - 1):.2%}",
                                '当前价格': f"{current_price:.2f}",
                                '涨停价': f"{high_limit:.2f}"
                            }
                            record_sell_trade(context, stock, "月初不涨停时间止损", details, current_data, date)
                            order_target_value(stock, 0)
                except Exception as e:
#                     log.error(f"{stock} 月初止损策略执行失败: {str(e)}")
                # 2. 低于昨日收盘价策略（需放量确认）
                    pass
                try:
                    price_df = get_price(
                        stock,
                        end_date=context.previous_date,
                        count=1,
                        fields=['close'],
                        skip_paused=False
                    )

                    if price_df is not None and not price_df.empty:
                        yesterday_close = price_df['close'].iloc[-1]

                        if not pd.isna(yesterday_close) and current_price < yesterday_close:
                            vol_ratio = get_5min_volume_ratio(stock, context, period=5)
                            if vol_ratio is not None and vol_ratio > g.max_sell_vol_ratio:
                                details = {
                                    '昨日收盘': f"{yesterday_close:.2f}",
                                    '当前价格': f"{current_price:.2f}",
                                    '跌幅': f"{(current_price / yesterday_close - 1):.2%}",
                                    '5分钟量比': f"{vol_ratio:.2f}"
                                }
                                record_sell_trade(context, stock, "低于昨日收盘价（放量确认）", details, current_data, date)
                                order_target_value(stock, 0)
                            else:
#                                 log.info(f"{stock} 低于昨日收盘价但未放量（量比{vol_ratio}），暂不卖出")
                                pass
                except Exception as e:
#                     log.error(f"{stock} 昨日收盘价策略执行失败: {str(e)}")

            # ==================== 下午14:50止盈策略 ====================
                    pass
            if is_profit_taking_time:
                try:
                    sdf = hist_map.get(stock)
                    if sdf is None or len(sdf) < 2:
#                         log.warning(f"{stock} 无法获取昨收，跳过止盈计算")
                        continue
                    yesterday_close = float(sdf['close'].iloc[-1])

                    # 安全检查：避免除零错误
                    if avg_cost == 0:
#                         log.warning(f"{stock} 平均成本为0，跳过止盈计算")
                        continue

                    # 止盈条件,量能过大：
                    # ========== 核心：重构 → get_bars(5min+include_now=True)汇总成交估算全天 ==========
                    # 【常量定义】A股交易规则+单位转换
                    UNIT_5MIN = 5  # 5分钟K线单位
                    TOTAL_TRADING_MINUTES = 240  # A股全天交易分钟数（9:30-11:30/13:00-15:00）
                    TOTAL_5MIN_BAR = TOTAL_TRADING_MINUTES // UNIT_5MIN  # 全天5分钟K线总数：48根（固定值）
                    HANDS_2_SHARE = 100  # 1手=100股
                    TEN_THOUSAND = 10000  # 万单位
                    SHARE_2_10KHAND = HANDS_2_SHARE * TEN_THOUSAND  # 股 → 万手 转换系数（除数）

                    # ---------------------- 步骤1：按要求调用get_bars获取5分钟成交列表 ----------------------
                    # 参数严格按你的要求：unit=5min + include_now=True + 取date/volume用于筛选和汇总
                    min5_bars = get_bars(
                        security=stock,
                        count=100,  # 取100根，远大于当日48根，确保覆盖当日所有5分钟K线
                        unit='5m',  # 按你的要求：单位5分钟
                        fields=('date', 'volume'),  # 取日期（筛选当日）+成交量（汇总）
                        include_now=True,  # 按你的要求：True=包含截止当前14:50的K线
                        end_dt=current_dt,  # 截止时间=14:50，精准定位
                        df=True  # 返回DataFrame，方便筛选/汇总
                    )

                    # ---------------------- 步骤2：数据基础校验 ----------------------
                    if min5_bars.empty or 'volume' not in min5_bars.columns or 'date' not in min5_bars.columns:
#                         log.warning(f"{stock} 5分钟K线获取失败（空数据/缺字段），跳过量能卖出")
                        continue
                    # 转换date为datetime格式，方便筛选当日数据
                    min5_bars['date'] = pd.to_datetime(min5_bars['date'])

                    # ---------------------- 步骤3：筛选【当日】的5分钟K线，避免混入前日数据 ----------------------
                    min5_bars_today = min5_bars[min5_bars['date'].dt.date == today_date].copy()
                    if min5_bars_today.empty:
#                         log.warning(f"{stock} 5分钟K线中无当日数据，跳过量能卖出")
                        continue

                    # ---------------------- 步骤4：汇总当日累计成交量（14:50真实成交） ----------------------
                    today_curr_vol_share = min5_bars_today['volume'].sum()  # 当日累计成交量（股）
                    curr_5min_bar_count = len(min5_bars_today)  # 当日已交易5分钟K线数
                    # 过滤异常成交量（累计成交为0/极少）
                    if today_curr_vol_share < 100:
#                         log.warning(f"{stock} 当日累计成交量异常（{today_curr_vol_share:.0f}股），跳过量能卖出")
                        continue

                    # ---------------------- 步骤5：获取【前日完整成交量】（精准无偏差） ----------------------
                    pre_day_vol_data = get_bars(
                        security=stock,
                        count=1,
                        unit='1d',
                        fields='volume',
                        include_now=False,  # False=仅取前日完整日线，无偏差
                        df=True
                    )
                    if pre_day_vol_data.empty:
#                         log.warning(f"{stock} 前日成交量获取失败，跳过量能卖出")
                        continue
                    pre_day_vol_share = pre_day_vol_data['volume'].iloc[0]  # 前日成交量（股）
                    if pre_day_vol_share < 100:
#                         log.warning(f"{stock} 前日成交量异常（{pre_day_vol_share:.0f}股），跳过量能卖出")
                        continue

                    # ---------------------- 步骤6：单位转换（股 → 万手），和你的实际数值对齐 ----------------------
                    pre_day_vol_10khand = pre_day_vol_share / SHARE_2_10KHAND  # 前日成交量（万手）
                    today_curr_vol_10khand = today_curr_vol_share / SHARE_2_10KHAND  # 当日累计成交（万手）

                    # ---------------------- 步骤7：估算全天成交量（按5分钟bar数，贴合交易节奏） ----------------------
                    # 估算系数=全天5分钟bar数/已交易bar数（14:50已交易约46-47根，系数≈1.02-1.04，误差极小）
                    estimate_coeff = TOTAL_5MIN_BAR / curr_5min_bar_count if curr_5min_bar_count > 0 else 1.0
                    # 估算全天成交量（股+万手）
                    today_est_vol_share = today_curr_vol_share * estimate_coeff
                    today_est_vol_10khand = today_curr_vol_10khand * estimate_coeff

                    # ---------------------- 步骤8：倍数计算（兜底取最大值，避免低估） ----------------------
                    vol_ratio_curr = today_curr_vol_share / pre_day_vol_share  # 累计倍数（当前/前日）
                    vol_ratio_est = today_est_vol_share / pre_day_vol_share  # 估算倍数（全天/前日）
                    final_vol_ratio = max(vol_ratio_curr, vol_ratio_est)  # 最终判定倍数（兜底取大）

                    # ---------------------- 步骤9：全量日志打印（精准排查，和你的实际数值对比） ----------------------
#                     log.info(f"📊 {stock} 14:50量能详情（5min K线精准汇总）：")
#                     log.info(f"   前日成交量：{pre_day_vol_share:,.0f}股 | {pre_day_vol_10khand:.2f}万手")
#                     log.info(f"   当日已交易5min K线：{curr_5min_bar_count}根 / 全天{TOTAL_5MIN_BAR}根")
#                     log.info(f"   当日累计成交：{today_curr_vol_share:,.0f}股 | {today_curr_vol_10khand:.2f}万手")
#                     log.info(f"   估算全天成交：{today_est_vol_share:,.0f}股 | {today_est_vol_10khand:.2f}万手")
#                     log.info(
#                         f"   估算系数：{estimate_coeff:.3f} | 累计倍数：{vol_ratio_curr:.2f} | 估算倍数：{vol_ratio_est:.2f} | 最终倍数：{final_vol_ratio:.2f}")

                    # ---------------------- 步骤10：触发卖出（未涨停 + 放量止盈双档） ----------------------
                    details_base = {
                        '成本价': f"{avg_cost:.2f}",
                        '当前价': f"{current_price:.2f}",
                        '前日成交量(万手)': f"{pre_day_vol_10khand:.2f}",
                        '当日累计成交(万手)': f"{today_curr_vol_10khand:.2f}",
                        '估算全天成交(万手)': f"{today_est_vol_10khand:.2f}",
                        '最终判定倍数': f"{final_vol_ratio:.2f}",
                        '涨停价': f"{high_limit:.2f}",
                        '是否涨停': '否'
                    }
                    if final_vol_ratio > 4:
                        record_sell_trade(context, stock, "尾盘放量4倍卖出(5min K线)", details_base, current_data, date)
                        order_target_value(stock, 0)
#                         log.info(f"📉 放量卖出 {stock}：最终倍数{final_vol_ratio:.2f}>4，未涨停，执行清仓！")
                        continue
                    else:
#                         log.info(f"📊 {stock} 未满足放量卖出条件（最终倍数{final_vol_ratio:.2f}/4，未涨停）")

                        pass
                except Exception as e:
#                     log.error(f"止盈逻辑执行出错 {stock}: {str(e)}")
                    continue

            # 下午时间段卖出策略
            if is_afternoon:
                # 安全检查：避免除零错误
                if avg_cost == 0:
#                     log.warning(f"{stock} 平均成本为0，跳过止损计算")
                    continue

                # 计算止损/回撤比例
                loss_pct = (avg_cost - current_price) / avg_cost
                high_limit_retreat = (high_limit - current_price) / avg_cost

                # ========== 完善：当前涨停则不止损，处理浮点数精度问题 ==========
                # 判定涨停：当前价格 >= 涨停价 - 1e-6（浮点数精度误差，避免0.0001的差价误判）
                is_limit_up = current_price >= high_limit - 1e-6
                if is_limit_up:
#                     log.info(f"{stock} 当前涨停（价:{current_price:.2f}/涨停:{high_limit:.2f}），跳过止损策略")
                    # 仅跳过止损逻辑，不影响后续MA5、量价顶背离等策略
                    pass
                # 未涨停时，才执行止损/涨停回撤卖出逻辑
                else:
                    if loss_pct >= 0.05 or high_limit_retreat >= 0.15:
                        details = {
                            '成本价': f"{avg_cost:.2f}",
                            '当前价': f"{current_price:.2f}",
                            '亏损比例': f"{loss_pct:.2%}",
                            '涨停回撤': f"{high_limit_retreat:.2%}"
                        }
                        record_sell_trade(context, stock, "止损卖出", details, current_data, date)
                        order_target_value(stock, 0)

                # 2. MA5均线策略（无论是否涨停，均正常执行）
                try:
                    sdf = hist_map.get(stock)
                    if sdf is not None and len(sdf) >= 4:
                        M4 = sdf['close'].mean()
                        MA5 = (M4 * 4 + current_price) / 5
                        if current_price < MA5:
                            details = {
                                'MA5': f"{MA5:.2f}",
                                '当前价': f"{current_price:.2f}",
                                '偏离率': f"{(current_price / MA5 - 1):.2%}"
                            }
                            record_sell_trade(context, stock, "跌破MA5均线", details, current_data, date)
                            order_target_value(stock, 0)
                except Exception as e:
#                     log.error(f"{stock} MA5策略执行失败: {str(e)}")

                # 4. 14:50尾盘：前日涨停+当日烂板放量+现价在均价线下方卖出
                    pass
                if is_profit_taking_time:
                    try:
                        sdf = hist_map.get(stock)
                        if sdf is not None and len(sdf) >= 2:
                            yday_close = float(sdf['close'].iloc[-1])
                            yday_high = float(sdf['high'].iloc[-1])
                            yday_vol = float(sdf['volume'].iloc[-1])
                            is_yday_limit_up = (yday_close >= yday_high * 0.995 and yday_high > 0)

                            if is_yday_limit_up and not is_limit_up and yday_vol > 0:
                                today_start = current_dt.replace(hour=9, minute=30, second=0)
                                min_data = get_price(stock, start_date=today_start, end_date=current_dt,
                                                     frequency='1m', fields=['high', 'volume', 'money'], skip_paused=True)
                                if not min_data.empty and min_data['volume'].sum() > 0:
                                    today_acc_vol = min_data['volume'].sum()
                                    today_acc_money = min_data['money'].sum()
                                    today_vwap = today_acc_money / today_acc_vol
                                    today_high = max(float(min_data['high'].max()), current_price)
                                    vol_ratio = today_acc_vol / yday_vol
                                    market_min = 120 + (current_time.hour - 13) * 60 + current_time.minute
                                    if market_min > 0:
                                        est_ratio = vol_ratio * (240.0 / market_min)
                                        sell_vol_ratio = getattr(g, 'limit_break_sell_vol_ratio', 1.8)
                                        vwap_discount = getattr(g, 'limit_break_sell_vwap_discount', 0.98)
                                        retreat_limit = getattr(g, 'limit_break_sell_intraday_retreat', 0.05)
                                        intraday_retreat = ((today_high - current_price) / today_high
                                                           if today_high > 0 else 0)
                                        weak_break_board = (
                                            current_price < yday_close or
                                            current_price < today_vwap * vwap_discount or
                                            intraday_retreat >= retreat_limit
                                        )
                                        if est_ratio > sell_vol_ratio and current_price < today_vwap and weak_break_board:
                                            details = {
                                                '触发信号': '前日涨停+烂板放量+破均价线',
                                                '前日收盘': f"{yday_close:.2f}(涨停)",
                                                '昨日成交(股)': f"{yday_vol:,.0f}",
                                                '当日累计(股)': f"{today_acc_vol:,.0f}",
                                                '当日均价': f"{today_vwap:.2f}",
                                                '估算全天倍数': f"{est_ratio:.2f}",
                                                '当前价': f"{current_price:.2f}",
                                                '日内最高': f"{today_high:.2f}",
                                                '日内回撤': f"{intraday_retreat:.2%}",
                                                '涨停价': f"{high_limit:.2f}",
                                                '均线偏离': f"{(current_price/today_vwap-1):.2%}",
                                                '走弱确认': '跌破昨收/深破均价/日内大回撤'
                                            }
                                            record_sell_trade(context, stock, "前日涨停烂板放量卖出", details, current_data, date)
                                            order_target_value(stock, 0)
#                                             log.info(f"🚨 {stock} 前日涨停烂板放量(est={est_ratio:.1f}x>1.8)且破均价(t_vwap={today_vwap:.2f})，尾盘卖出防利润回吐")
                                            continue
                    except Exception as e:
#                         log.error(f"{stock} 前日涨停烂板检测执行失败: {str(e)}")

            # 3. 新增：最近24根半小时K线量价顶背离策略（全局策略，无论上下午/是否涨停均执行）
                        pass
            try:
                # 获取最近24根半小时K线数据（包含最高价和成交量）
                # frequency='30m'表示半小时K线，count=24获取最近24根
                kline_data = get_price(
                    stock,
                    end_date=context.current_dt,
                    count=24,
                    frequency='30m',
                    fields=['high', 'volume'],
                    skip_paused=False
                )

                # 数据有效性校验
                if kline_data is None or kline_data.empty or len(kline_data) < 24:
#                     log.warning(
#                         f"{stock} 无法获取足够的24根半小时K线数据（实际获取{len(kline_data) if kline_data is not None else 0}根），跳过量价顶背离检查")
                    continue

                # 提取24根K线的最高价和成交量
                highs = kline_data['high'].values  # 最高价数组
                volumes = kline_data['volume'].values  # 成交量数组

                # 计算24根K线中的最高价格和最大成交量
                max_high = max(highs)
                max_volume = max(volumes)

                # 防护：避免最大成交量为0导致的计算问题
                if max_volume <= 0:
#                     log.warning(f"{stock} 最近24根半小时K线最大成交量为0，跳过量价顶背离检查")
                    continue

                # 最近一根K线的最高价和成交量
                last_kline_high = highs[-1]
                last_kline_volume = volumes[-1]

                # 计算当前价格与最近K线最高价的差距百分比
                price_drop_percent = (last_kline_high - current_price) / last_kline_high * 100

                # 量价顶背离条件：
                # 1. 最近一根K线最高价创24根内新高（等于最大最高价）
                # 2. 最近一根K线成交量 <= 最大成交量的一半
                # 3. 当前价格未涨停
                # 4. 新增条件：当前价格比最近一根K线最高价下跌超过3%
                if ((last_kline_high >= max_high - 1e-6) and
                        (last_kline_volume <= max_volume * 0.5) and
                        (current_price < high_limit) and
                        (price_drop_percent > 3.0)):  # 新增条件：价格下跌超过3%

                    details = {
                        '24根K线最高': f"{max_high:.2f}",
                        '最近K线最高': f"{last_kline_high:.2f}",
                        '24根最大量能': f"{max_volume:.0f}",
                        '最近K线量能': f"{last_kline_volume:.0f}",
                        '量能比例': f"{(last_kline_volume / max_volume):.2%}",
                        '当前价': f"{current_price:.2f}",
                        '价格回撤': f"{price_drop_percent:.2f}%",  # 新增：显示价格回撤百分比
                        '涨停价': f"{high_limit:.2f}"
                    }
                    record_sell_trade(context, stock, "24根半小时K线量价顶背离", details, current_data, date)
                    order_target_value(stock, 0)

                # 新增：当价格已经下跌超过5%时，也执行卖出操作，但使用不同的原因
                elif ((last_kline_high >= max_high - 1e-6) and
                      (last_kline_volume <= max_volume * 0.5) and
                      (price_drop_percent > 5.0)):

                    pass
                    details = {
                        '24根K线最高': f"{max_high:.2f}",
                        '最近K线最高': f"{last_kline_high:.2f}",
                        '24根最大量能': f"{max_volume:.0f}",
                        '最近K线量能': f"{last_kline_volume:.0f}",
                        '量能比例': f"{(last_kline_volume / max_volume):.2%}",
                        '当前价': f"{current_price:.2f}",
                        '价格回撤': f"{price_drop_percent:.2f}%",
                        '涨停价': f"{high_limit:.2f}"
                    }
                    record_sell_trade(context, stock, "24根半小时K线量价顶背离(价格已回撤超5%)", details, current_data, date)
                    order_target_value(stock, 0)

            except Exception as e:
#                 log.error(f"{stock} 量价顶背离策略执行失败: {str(e)}")


                pass
        except Exception as e:
#             log.error(f"处理 {stock} 卖出策略时发生错误: {str(e)}")


# 计算股票处于一段时间内相对位置
            pass
def get_relative_position_df(stock_list, date, watch_days):
    if len(stock_list) != 0:
        df = get_price(stock_list, end_date=date, fields=['high', 'low', 'close'], count=watch_days, fill_paused=False,
                       skip_paused=False, panel=False).dropna()
        close = df.groupby('code').apply(lambda df: df.iloc[-1, -1])
        high = df.groupby('code').apply(lambda df: df['high'].max())
        low = df.groupby('code').apply(lambda df: df['low'].min())
        result = pd.DataFrame()
        result['rp'] = (close - low) / (high - low)
        return result
    else:
        return pd.DataFrame(columns=['rp'])

    # 连板龙头函数


# 筛选按因子值排名的股票
def get_factor_filter_df(context, stock_list, jqfactor, sort):
    if len(stock_list) != 0:
        yesterday = context.previous_date
        score_list = get_factor_values(stock_list, jqfactor, end_date=yesterday, count=1)[jqfactor].iloc[0].tolist()
        df = pd.DataFrame(index=stock_list, data={'score': score_list}).dropna()
        df = df.sort_values(by='score', ascending=sort)
    else:
        df = pd.DataFrame(index=[], data={'score': []})
    return df


# 概念筛选
def filter_concept_stock(dct, concept):
    tmp_set = set()
    for k, v in dct.items():
        for d in dct[k]['jq_concept']:
            if d['concept_name'] == concept:
                tmp_set.add(k)
    return list(tmp_set)


# 计算热门概念
def get_hot_concept(dct, date):
    # 计算出现涨停最多的概念
    concept_count = {}
    for key in dct:
        for i in dct[key]['jq_concept']:
            if i['concept_name'] in concept_count.keys():
                concept_count[i['concept_name']] += 1
            else:
                if i['concept_name'] not in ['转融券标的', '融资融券', '深股通', '沪股通']:
                    concept_count[i['concept_name']] = 1
    df = pd.DataFrame(list(concept_count.items()), columns=['concept_name', 'concept_count'])
    df = df.set_index('concept_name')
    df = df.sort_values(by='concept_count', ascending=False)
    max_num = df.iloc[0, 0]
    df = df[df['concept_count'] == max_num]
    concept = list(df.index)[0]
    return concept


# 反向首板低开函数
# 筛选出某一日涨停的股票
def get_ll_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close', 'low_limit'], count=1, panel=False,
                   fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    df = df[df['close'] == df['low_limit']]
    hl_list = list(df.code)
    return hl_list


# 计算涨停数
def get_ll_count_df(hl_list, date, watch_days):
    # 获取watch_days的数据
    df = get_price(hl_list, end_date=date, frequency='daily', fields=['low', 'close', 'low_limit'], count=watch_days,
                   panel=False, fill_paused=False, skip_paused=False)
    df.index = df.code
    # 计算涨停与一字涨停数，一字涨停定义为最低价等于涨停价
    hl_count_list = []
    extreme_hl_count_list = []
    for stock in hl_list:
        df_sub = df.loc[stock]
        hl_days = df_sub[df_sub.close == df_sub.low_limit].low_limit.count()
        extreme_hl_days = df_sub[df_sub.low == df_sub.low_limit].low_limit.count()
        hl_count_list.append(hl_days)
        extreme_hl_count_list.append(extreme_hl_days)
    # 创建df记录
    df = pd.DataFrame(index=hl_list, data={'count': hl_count_list, 'extreme_count': extreme_hl_count_list})
    return df


# 计算连板数
def get_continue_count_df_ll(hl_list, date, watch_days):
    df = pd.DataFrame()
    for d in range(2, watch_days + 1):
        HLC = get_ll_count_df(hl_list, date, d)
        CHLC = HLC[HLC['count'] == d]
        df = df.append(CHLC)
    stock_list = list(set(df.index))
    ccd = pd.DataFrame()
    for s in stock_list:
        tmp = df.loc[[s]]
        if len(tmp) > 1:
            M = tmp['count'].max()
            tmp = tmp[tmp['count'] == M]
        ccd = ccd.append(tmp)
    if len(ccd) != 0:
        ccd = ccd.sort_values(by='count', ascending=False)
    return ccd
