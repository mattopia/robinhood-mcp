"""Read-only Robinhood tools wrapping robin_stocks library."""

from typing import Any, Literal

import robin_stocks.robinhood as rh
from robin_stocks.robinhood.helper import SESSION


class RobinhoodError(Exception):
    """Error from Robinhood API call."""

    pass


def _safe_call(func: callable, *args, **kwargs) -> Any:
    """Safely call a robin_stocks function with error handling.

    Args:
        func: The robin_stocks function to call.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        The function result.

    Raises:
        RobinhoodError: If the call fails.
    """
    try:
        result = func(*args, **kwargs)
        if result is None:
            raise RobinhoodError("API returned None - you may need to login first")
        return result
    except RobinhoodError:
        raise
    except Exception as e:
        raise RobinhoodError(f"API call failed: {e}") from e


_ACCOUNTS_URL = (
    "https://api.robinhood.com/accounts/"
    "?default_to_all_accounts=true"
    "&include_managed=true"
    "&include_multiple_individual=true"
    "&include_multiple_individual_accounts=true"
    "&is_default=false"
)

_ACCOUNT_FIELDS = (
    "account_number",
    "type",
    "brokerage_account_type",
    "cash",
    "buying_power",
    "is_default",
    "created_at",
)


def get_accounts() -> list[dict[str, Any]]:
    """Get all Robinhood accounts for the logged-in user.

    Uses extended query parameters to surface IRA accounts alongside the
    standard brokerage account. Distinguish account types by the
    brokerage_account_type field:
      - "individual"        → standard brokerage
      - "ira_traditional"   → Traditional IRA
      - "ira_roth"          → Roth IRA

    Returns:
        List of accounts, each with account_number, type, brokerage_account_type,
        cash, buying_power, is_default, is_primary_account, and created_at.
    """
    try:
        resp = SESSION.get(_ACCOUNTS_URL)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RobinhoodError(f"Failed to fetch accounts: {e}") from e

    accounts = data.get("results", [])
    result = []
    for acct in accounts:
        if not acct:
            continue
        entry = {k: acct.get(k) for k in _ACCOUNT_FIELDS}
        # is_primary_account is nested: margin_balances for brokerage, cash_balances for IRA
        balances = acct.get("margin_balances") or acct.get("cash_balances") or {}
        entry["is_primary_account"] = balances.get("is_primary_account")
        result.append(entry)
    return result


def get_portfolio(account_number: str | None = None) -> dict[str, Any]:
    """Get current portfolio value and performance metrics.

    Args:
        account_number: Specific account number to query (from get_accounts()).
            If None, returns the primary brokerage account.

    Returns:
        Portfolio profile with equity, extended hours equity, market value, etc.
    """
    return _safe_call(rh.profiles.load_portfolio_profile, account_number=account_number)


def _enrich_positions(raw_positions: list[dict]) -> dict[str, dict[str, Any]]:
    """Enrich raw position records to match the build_holdings() output format.

    Resolves instrument URLs to symbols, batch-fetches current quotes and
    fundamentals, then computes equity, percent_change, equity_change, and
    portfolio percentage — the same fields build_holdings() returns for the
    default brokerage account.
    """
    # Step 1: resolve instrument URLs → symbols and names
    interim: dict[str, dict] = {}
    for pos in raw_positions:
        if not pos:
            continue
        try:
            instrument = rh.stocks.get_instrument_by_url(pos["instrument"])
            symbol = instrument["symbol"]
            interim[symbol] = {
                "quantity": pos["quantity"],
                "average_buy_price": pos["average_buy_price"],
                "name": instrument.get("simple_name") or instrument.get("name", ""),
                "id": instrument.get("id", ""),
                "type": "stock",
            }
        except Exception:
            continue

    if not interim:
        return {}

    symbols = list(interim.keys())

    # Step 2: batch-fetch current quotes
    price_map: dict[str, str] = {}
    try:
        quotes = rh.stocks.get_quotes(symbols) or []
        for q in quotes:
            if q and q.get("symbol"):
                price_map[q["symbol"]] = q.get("last_trade_price") or q.get(
                    "last_extended_hours_trade_price"
                )
    except Exception:
        pass

    # Step 3: batch-fetch fundamentals for pe_ratio
    pe_map: dict[str, str | None] = {}
    try:
        funds = rh.stocks.get_fundamentals(symbols) or []
        for symbol, f in zip(symbols, funds):
            pe_map[symbol] = f.get("pe_ratio") if f else None
    except Exception:
        pass

    # Step 4: compute derived fields; accumulate total equity for percentage
    total_equity = 0.0
    computed: dict[str, dict] = {}
    for symbol, data in interim.items():
        price_str = price_map.get(symbol)
        avg_str = data["average_buy_price"]
        qty_str = data["quantity"]
        try:
            price = float(price_str) if price_str else None
            avg = float(avg_str) if avg_str else None
            qty = float(qty_str) if qty_str else 0.0
        except (ValueError, TypeError):
            price = avg = None
            qty = 0.0

        equity = price * qty if price is not None else None
        if equity:
            total_equity += equity

        percent_change = None
        equity_change = None
        if price is not None and avg and avg != 0:
            percent_change = (price - avg) / avg * 100
            equity_change = (price - avg) * qty

        computed[symbol] = {
            "price": str(price) if price is not None else None,
            "quantity": qty_str,
            "average_buy_price": avg_str,
            "equity": str(equity) if equity is not None else None,
            "percent_change": str(round(percent_change, 2)) if percent_change is not None else None,
            "equity_change": str(round(equity_change, 2)) if equity_change is not None else None,
            "type": "stock",
            "name": data["name"],
            "id": data["id"],
            "pe_ratio": pe_map.get(symbol),
            "percentage": None,  # filled in below
        }

    # Step 5: fill in portfolio percentage now that total_equity is known
    for symbol, data in computed.items():
        try:
            eq = float(data["equity"]) if data["equity"] else 0.0
            pct = round(eq / total_equity * 100, 2) if total_equity > 0 else 0.0
            data["percentage"] = str(pct)
        except (ValueError, TypeError):
            data["percentage"] = "0.00"

    return computed


def get_positions(account_number: str | None = None) -> dict[str, dict[str, Any]]:
    """Get all current stock positions with details.

    Args:
        account_number: Specific account number to query (from get_accounts()).
            If None, returns enriched holdings for the primary brokerage account.

    Returns:
        Dict mapping symbol to position details including price, quantity,
        average_buy_price, equity, percent_change, equity_change, name,
        pe_ratio, and percentage.
    """
    if account_number is None:
        # build_holdings enriches positions with current price, equity, % change, etc.
        return _safe_call(rh.account.build_holdings)

    # build_holdings doesn't support account_number; use get_open_stock_positions
    # and enrich the results ourselves to produce the same field set.
    positions = _safe_call(rh.account.get_open_stock_positions, account_number=account_number)
    if not isinstance(positions, list):
        return {}
    return _enrich_positions(positions)


def get_watchlist(name: str = "Default") -> list[dict[str, Any]]:
    """Get stocks in a watchlist.

    Args:
        name: Watchlist name (default: "Default").

    Returns:
        List of watchlist items with instrument details.
    """
    result = _safe_call(rh.account.get_watchlist_by_name, name=name)
    return result if isinstance(result, list) else []


def get_quote(symbol: str) -> dict[str, Any]:
    """Get real-time quote for a stock symbol.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL").

    Returns:
        Quote data including last_trade_price, bid, ask, etc.
    """
    if not symbol or not isinstance(symbol, str):
        raise RobinhoodError("Symbol must be a non-empty string")

    symbol = symbol.upper().strip()
    result = _safe_call(rh.stocks.get_quotes, symbol)

    if isinstance(result, list) and len(result) > 0:
        return result[0]
    raise RobinhoodError(f"No quote found for symbol: {symbol}")


def get_fundamentals(symbol: str) -> dict[str, Any]:
    """Get fundamental data for a stock.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Fundamentals including pe_ratio, market_cap, dividend_yield, etc.
    """
    if not symbol or not isinstance(symbol, str):
        raise RobinhoodError("Symbol must be a non-empty string")

    symbol = symbol.upper().strip()
    result = _safe_call(rh.stocks.get_fundamentals, symbol)

    if isinstance(result, list) and len(result) > 0:
        return result[0]
    raise RobinhoodError(f"No fundamentals found for symbol: {symbol}")


def get_historicals(
    symbol: str,
    interval: Literal["5minute", "10minute", "hour", "day", "week"] = "day",
    span: Literal["day", "week", "month", "3month", "year", "5year"] = "month",
) -> list[dict[str, Any]]:
    """Get historical price data for a stock.

    Args:
        symbol: Stock ticker symbol.
        interval: Time interval (5minute, 10minute, hour, day, week).
        span: Time span (day, week, month, 3month, year, 5year).

    Returns:
        List of historical data points with open, close, high, low, volume.
    """
    if not symbol or not isinstance(symbol, str):
        raise RobinhoodError("Symbol must be a non-empty string")

    symbol = symbol.upper().strip()

    valid_intervals = {"5minute", "10minute", "hour", "day", "week"}
    valid_spans = {"day", "week", "month", "3month", "year", "5year"}

    if interval not in valid_intervals:
        raise RobinhoodError(f"Invalid interval. Must be one of: {valid_intervals}")
    if span not in valid_spans:
        raise RobinhoodError(f"Invalid span. Must be one of: {valid_spans}")

    result = _safe_call(rh.stocks.get_stock_historicals, symbol, interval=interval, span=span)
    return result if isinstance(result, list) else []


def get_news(symbol: str) -> list[dict[str, Any]]:
    """Get recent news articles for a stock.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        List of news articles with title, url, source, published_at, etc.
    """
    if not symbol or not isinstance(symbol, str):
        raise RobinhoodError("Symbol must be a non-empty string")

    symbol = symbol.upper().strip()
    result = _safe_call(rh.stocks.get_news, symbol)
    return result if isinstance(result, list) else []


def get_earnings(symbol: str) -> list[dict[str, Any]]:
    """Get earnings data for a stock.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        List of earnings reports with eps, report date, estimates, etc.
    """
    if not symbol or not isinstance(symbol, str):
        raise RobinhoodError("Symbol must be a non-empty string")

    symbol = symbol.upper().strip()
    result = _safe_call(rh.stocks.get_earnings, symbol)
    return result if isinstance(result, list) else []


def get_ratings(symbol: str) -> dict[str, Any]:
    """Get analyst ratings summary for a stock.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Ratings summary with buy, hold, sell counts and summary.
    """
    if not symbol or not isinstance(symbol, str):
        raise RobinhoodError("Symbol must be a non-empty string")

    symbol = symbol.upper().strip()
    result = _safe_call(rh.stocks.get_ratings, symbol)

    if isinstance(result, dict):
        return result
    raise RobinhoodError(f"No ratings found for symbol: {symbol}")


def get_dividends(account_number: str | None = None) -> list[dict[str, Any]]:
    """Get all dividend payments received.

    Args:
        account_number: Specific account number to filter by (from get_accounts()).
            If None, returns dividends across all accounts.
            If provided, filters to only dividends for that account.

    Returns:
        List of dividend payments with amount, payable_date, record_date, etc.
    """
    # The Robinhood API doesn't support account_number filtering on the dividends endpoint,
    # so we fetch all and filter locally. Each dividend record contains an 'account' URL
    # like "https://api.robinhood.com/accounts/{account_number}/".
    result = _safe_call(rh.account.get_dividends)
    if not isinstance(result, list):
        return []
    if account_number is None:
        return result
    return [d for d in result if account_number in d.get("account", "")]


def get_options_positions() -> list[dict[str, Any]]:
    """Get all current options positions.

    Returns:
        List of options positions with chain_symbol, type, quantity, etc.
    """
    result = _safe_call(rh.options.get_open_option_positions)
    return result if isinstance(result, list) else []


def get_crypto_positions() -> list[dict[str, Any]]:
    """Get all current crypto positions.

    Returns:
        List of crypto positions with currency_code, quantity, average_buy_price, etc.
    """
    result = _safe_call(rh.crypto.get_crypto_positions)
    return result if isinstance(result, list) else []


def get_crypto_quote(symbol: str) -> dict[str, Any]:
    """Get real-time quote for a crypto currency.

    Args:
        symbol: Crypto symbol (e.g., "BTC", "ETH", "DOGE").

    Returns:
        Quote data including ask_price, bid_price, mark_price, etc.
    """
    if not symbol or not isinstance(symbol, str):
        raise RobinhoodError("Symbol must be a non-empty string")

    symbol = symbol.upper().strip()
    result = _safe_call(rh.crypto.get_crypto_quote, symbol)

    if isinstance(result, dict):
        return result
    raise RobinhoodError(f"No crypto quote found for symbol: {symbol}")


def get_crypto_historicals(
    symbol: str,
    interval: Literal["15second", "5minute", "10minute", "hour", "day", "week"] = "day",
    span: Literal["hour", "day", "week", "month", "3month", "year", "5year"] = "week",
) -> list[dict[str, Any]]:
    """Get historical price data for a crypto currency.

    Args:
        symbol: Crypto symbol (e.g., "BTC", "ETH").
        interval: Time interval (15second, 5minute, 10minute, hour, day, week).
        span: Time span (hour, day, week, month, 3month, year, 5year).

    Returns:
        List of historical data points with open, close, high, low, volume.
    """
    if not symbol or not isinstance(symbol, str):
        raise RobinhoodError("Symbol must be a non-empty string")

    symbol = symbol.upper().strip()

    valid_intervals = {"15second", "5minute", "10minute", "hour", "day", "week"}
    valid_spans = {"hour", "day", "week", "month", "3month", "year", "5year"}

    if interval not in valid_intervals:
        raise RobinhoodError(f"Invalid interval. Must be one of: {valid_intervals}")
    if span not in valid_spans:
        raise RobinhoodError(f"Invalid span. Must be one of: {valid_spans}")

    result = _safe_call(rh.crypto.get_crypto_historicals, symbol, interval=interval, span=span)
    return result if isinstance(result, list) else []


def search_symbols(query: str) -> list[dict[str, Any]]:
    """Search for stock symbols by company name or ticker.

    Args:
        query: Search query (company name or partial ticker).

    Returns:
        List of matching instruments with symbol, name, etc.
    """
    if not query or not isinstance(query, str):
        raise RobinhoodError("Query must be a non-empty string")

    query = query.strip()

    # Try to get instruments by the query
    try:
        result = rh.stocks.get_instruments_by_symbols(query.upper())
        if result and isinstance(result, list):
            return result
    except Exception:
        pass

    # If exact match fails, try search
    try:
        result = rh.stocks.find_instrument_data(query)
        return result if isinstance(result, list) else []
    except Exception as e:
        raise RobinhoodError(f"Search failed: {e}") from e
