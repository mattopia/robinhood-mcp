"""Tests for tools module."""

from unittest.mock import MagicMock, patch

import pytest

from robinhood_mcp.tools import (
    RobinhoodError,
    get_accounts,
    get_crypto_historicals,
    get_crypto_positions,
    get_crypto_quote,
    get_dividends,
    get_earnings,
    get_fundamentals,
    get_historicals,
    get_news,
    get_options_positions,
    get_portfolio,
    get_positions,
    get_quote,
    get_ratings,
    get_watchlist,
    search_symbols,
)


class TestGetAccounts:
    """Tests for get_accounts function."""

    def _make_account(self, account_number, brokerage_account_type, is_primary, use_margin=True):
        acct = {
            "account_number": account_number,
            "type": "margin" if use_margin else "cash",
            "brokerage_account_type": brokerage_account_type,
            "cash": "1000.00",
            "buying_power": "2000.00",
            "is_default": is_primary,
            "created_at": "2020-01-01T00:00:00Z",
        }
        # is_primary_account lives in margin_balances for brokerage, cash_balances for IRA
        if use_margin:
            acct["margin_balances"] = {"is_primary_account": is_primary}
        else:
            acct["cash_balances"] = {"is_primary_account": is_primary}
        return acct

    @patch("robinhood_mcp.tools.SESSION")
    def test_returns_all_accounts(self, mock_session: MagicMock):
        """Should return brokerage and IRA accounts using the extended endpoint."""
        api_response = {
            "results": [
                self._make_account("ABC123", "individual", True),
                self._make_account("IRA456", "ira_traditional", False),
            ]
        }
        mock_session.get.return_value.json.return_value = api_response
        mock_session.get.return_value.raise_for_status = MagicMock()

        result = get_accounts()

        assert len(result) == 2
        assert result[0]["account_number"] == "ABC123"
        assert result[0]["brokerage_account_type"] == "individual"
        assert result[1]["account_number"] == "IRA456"
        assert result[1]["brokerage_account_type"] == "ira_traditional"

    @patch("robinhood_mcp.tools.SESSION")
    def test_extracts_is_primary_account_from_balances(self, mock_session: MagicMock):
        """is_primary_account is in margin_balances for brokerage, cash_balances for IRA."""
        api_response = {
            "results": [
                self._make_account("ABC123", "individual", True, use_margin=True),
                self._make_account("IRA456", "ira_traditional", False, use_margin=False),
            ]
        }
        mock_session.get.return_value.json.return_value = api_response
        mock_session.get.return_value.raise_for_status = MagicMock()

        result = get_accounts()

        assert result[0]["is_primary_account"] is True
        assert result[1]["is_primary_account"] is False

    @patch("robinhood_mcp.tools.SESSION")
    def test_returns_only_known_fields(self, mock_session: MagicMock):
        """Should return only the declared fields, not raw API noise."""
        acct = self._make_account("ABC123", "individual", True)
        acct["internal_field"] = "should_not_appear"
        mock_session.get.return_value.json.return_value = {"results": [acct]}
        mock_session.get.return_value.raise_for_status = MagicMock()

        result = get_accounts()

        assert "internal_field" not in result[0]

    @patch("robinhood_mcp.tools.SESSION")
    def test_raises_on_request_error(self, mock_session: MagicMock):
        """Should raise RobinhoodError when the HTTP request fails."""
        mock_session.get.side_effect = Exception("network error")

        with pytest.raises(RobinhoodError, match="Failed to fetch accounts"):
            get_accounts()


class TestGetPortfolio:
    """Tests for get_portfolio function."""

    @patch("robinhood_mcp.tools.rh.profiles.load_portfolio_profile")
    def test_returns_portfolio_data(self, mock_profile: MagicMock):
        """Should return portfolio profile data."""
        expected = {
            "equity": "10000.00",
            "extended_hours_equity": "10050.00",
        }
        mock_profile.return_value = expected

        result = get_portfolio()

        assert result == expected
        mock_profile.assert_called_once_with(account_number=None)

    @patch("robinhood_mcp.tools.rh.profiles.load_portfolio_profile")
    def test_passes_account_number(self, mock_profile: MagicMock):
        """Should pass account_number to the API call."""
        mock_profile.return_value = {"equity": "5000.00"}

        get_portfolio(account_number="IRA456")

        mock_profile.assert_called_once_with(account_number="IRA456")

    @patch("robinhood_mcp.tools.rh.profiles.load_portfolio_profile")
    def test_raises_on_none_result(self, mock_profile: MagicMock):
        """Should raise RobinhoodError when API returns None."""
        mock_profile.return_value = None

        with pytest.raises(RobinhoodError) as exc_info:
            get_portfolio()
        assert "login" in str(exc_info.value).lower()


class TestGetPositions:
    """Tests for get_positions function."""

    @patch("robinhood_mcp.tools.rh.account.build_holdings")
    def test_returns_holdings_for_default_account(self, mock_holdings: MagicMock):
        """Should use build_holdings (enriched data) when no account_number given."""
        expected = {
            "AAPL": {"quantity": "10", "average_buy_price": "150.00"},
            "TSLA": {"quantity": "5", "average_buy_price": "200.00"},
        }
        mock_holdings.return_value = expected

        result = get_positions()

        assert result == expected
        mock_holdings.assert_called_once_with()

    @patch("robinhood_mcp.tools.rh.stocks.get_fundamentals")
    @patch("robinhood_mcp.tools.rh.stocks.get_quotes")
    @patch("robinhood_mcp.tools.rh.stocks.get_instrument_by_url")
    @patch("robinhood_mcp.tools.rh.account.get_open_stock_positions")
    def test_enriches_positions_for_account_number(
        self,
        mock_positions: MagicMock,
        mock_instrument: MagicMock,
        mock_quotes: MagicMock,
        mock_fundamentals: MagicMock,
    ):
        """Should enrich IRA positions with the same fields as build_holdings."""
        mock_positions.return_value = [
            {
                "instrument": "https://api.robinhood.com/instruments/abc/",
                "quantity": "10",
                "average_buy_price": "100.00",
                "account_number": "IRA456",
            }
        ]
        mock_instrument.return_value = {
            "symbol": "VTI",
            "name": "Vanguard Total Stock",
            "id": "abc",
        }
        mock_quotes.return_value = [{"symbol": "VTI", "last_trade_price": "110.00"}]
        mock_fundamentals.return_value = [{"pe_ratio": "22.5"}]

        result = get_positions(account_number="IRA456")

        assert "VTI" in result
        vti = result["VTI"]
        assert vti["quantity"] == "10"
        assert vti["average_buy_price"] == "100.00"
        assert vti["price"] == "110.0"
        assert vti["equity"] == "1100.0"
        assert vti["pe_ratio"] == "22.5"
        assert vti["name"] == "Vanguard Total Stock"
        assert vti["percentage"] == "100.0"
        # percent_change: (110 - 100) / 100 * 100 = 10.0
        assert vti["percent_change"] == "10.0"
        # equity_change: (110 - 100) * 10 = 100.0
        assert vti["equity_change"] == "100.0"
        mock_positions.assert_called_once_with(account_number="IRA456")


class TestGetQuote:
    """Tests for get_quote function."""

    @patch("robinhood_mcp.tools.rh.stocks.get_quotes")
    def test_returns_quote(self, mock_quotes: MagicMock):
        """Should return quote for symbol."""
        expected = {"symbol": "AAPL", "last_trade_price": "175.00"}
        mock_quotes.return_value = [expected]

        result = get_quote("AAPL")

        assert result == expected
        mock_quotes.assert_called_once_with("AAPL")

    @patch("robinhood_mcp.tools.rh.stocks.get_quotes")
    def test_uppercases_symbol(self, mock_quotes: MagicMock):
        """Should uppercase and strip symbol."""
        mock_quotes.return_value = [{"symbol": "AAPL"}]

        get_quote("  aapl  ")

        mock_quotes.assert_called_once_with("AAPL")

    def test_raises_for_empty_symbol(self):
        """Should raise RobinhoodError for empty symbol."""
        with pytest.raises(RobinhoodError) as exc_info:
            get_quote("")
        assert "non-empty string" in str(exc_info.value)

    def test_raises_for_non_string_symbol(self):
        """Should raise RobinhoodError for non-string symbol."""
        with pytest.raises(RobinhoodError):
            get_quote(123)  # type: ignore

    @patch("robinhood_mcp.tools.rh.stocks.get_quotes")
    def test_raises_for_no_results(self, mock_quotes: MagicMock):
        """Should raise RobinhoodError when no quote found."""
        mock_quotes.return_value = []

        with pytest.raises(RobinhoodError) as exc_info:
            get_quote("INVALID")
        assert "No quote found" in str(exc_info.value)


class TestGetHistoricals:
    """Tests for get_historicals function."""

    @patch("robinhood_mcp.tools.rh.stocks.get_stock_historicals")
    def test_returns_historical_data(self, mock_hist: MagicMock):
        """Should return historical data."""
        expected = [
            {"open": "100.00", "close": "105.00", "volume": "1000000"},
            {"open": "105.00", "close": "110.00", "volume": "1200000"},
        ]
        mock_hist.return_value = expected

        result = get_historicals("AAPL", interval="day", span="month")

        assert result == expected
        mock_hist.assert_called_once_with("AAPL", interval="day", span="month")

    def test_raises_for_invalid_interval(self):
        """Should raise RobinhoodError for invalid interval."""
        with pytest.raises(RobinhoodError) as exc_info:
            get_historicals("AAPL", interval="invalid")  # type: ignore
        assert "Invalid interval" in str(exc_info.value)

    def test_raises_for_invalid_span(self):
        """Should raise RobinhoodError for invalid span."""
        with pytest.raises(RobinhoodError) as exc_info:
            get_historicals("AAPL", span="invalid")  # type: ignore
        assert "Invalid span" in str(exc_info.value)


class TestGetFundamentals:
    """Tests for get_fundamentals function."""

    @patch("robinhood_mcp.tools.rh.stocks.get_fundamentals")
    def test_returns_fundamentals(self, mock_fund: MagicMock):
        """Should return fundamental data."""
        expected = {"pe_ratio": "25.5", "market_cap": "2500000000000"}
        mock_fund.return_value = [expected]

        result = get_fundamentals("AAPL")

        assert result == expected


class TestGetNews:
    """Tests for get_news function."""

    @patch("robinhood_mcp.tools.rh.stocks.get_news")
    def test_returns_news_list(self, mock_news: MagicMock):
        """Should return list of news articles."""
        expected = [
            {"title": "Apple announces new product", "source": "Reuters"},
            {"title": "AAPL stock rises", "source": "Bloomberg"},
        ]
        mock_news.return_value = expected

        result = get_news("AAPL")

        assert result == expected


class TestGetEarnings:
    """Tests for get_earnings function."""

    @patch("robinhood_mcp.tools.rh.stocks.get_earnings")
    def test_returns_earnings_list(self, mock_earnings: MagicMock):
        """Should return list of earnings reports."""
        expected = [{"year": "2024", "quarter": "Q4", "eps": {"actual": "1.50"}}]
        mock_earnings.return_value = expected

        result = get_earnings("AAPL")

        assert result == expected


class TestGetRatings:
    """Tests for get_ratings function."""

    @patch("robinhood_mcp.tools.rh.stocks.get_ratings")
    def test_returns_ratings(self, mock_ratings: MagicMock):
        """Should return ratings summary."""
        expected = {"num_buy_ratings": 30, "num_hold_ratings": 10, "num_sell_ratings": 2}
        mock_ratings.return_value = expected

        result = get_ratings("AAPL")

        assert result == expected


class TestGetDividends:
    """Tests for get_dividends function."""

    @patch("robinhood_mcp.tools.rh.account.get_dividends")
    def test_returns_all_dividends_when_no_account_number(self, mock_divs: MagicMock):
        """Should return all dividends when no account_number given."""
        expected = [
            {
                "amount": "0.23",
                "payable_date": "2024-02-15",
                "account": "https://api.robinhood.com/accounts/ABC123/",
            },
            {
                "amount": "0.10",
                "payable_date": "2024-03-01",
                "account": "https://api.robinhood.com/accounts/IRA456/",
            },
        ]
        mock_divs.return_value = expected

        result = get_dividends()

        assert result == expected

    @patch("robinhood_mcp.tools.rh.account.get_dividends")
    def test_filters_dividends_by_account_number(self, mock_divs: MagicMock):
        """Should filter dividends by account_number using the account URL field."""
        all_divs = [
            {"amount": "0.23", "account": "https://api.robinhood.com/accounts/ABC123/"},
            {"amount": "0.10", "account": "https://api.robinhood.com/accounts/IRA456/"},
        ]
        mock_divs.return_value = all_divs

        result = get_dividends(account_number="IRA456")

        assert len(result) == 1
        assert result[0]["amount"] == "0.10"


class TestGetOptionsPositions:
    """Tests for get_options_positions function."""

    @patch("robinhood_mcp.tools.rh.options.get_open_option_positions")
    def test_returns_options_list(self, mock_options: MagicMock):
        """Should return list of options positions."""
        expected = [{"chain_symbol": "AAPL", "type": "call", "quantity": "1"}]
        mock_options.return_value = expected

        result = get_options_positions()

        assert result == expected


class TestGetWatchlist:
    """Tests for get_watchlist function."""

    @patch("robinhood_mcp.tools.rh.account.get_watchlist_by_name")
    def test_returns_watchlist(self, mock_watchlist: MagicMock):
        """Should return watchlist items."""
        expected = [{"symbol": "AAPL"}, {"symbol": "TSLA"}]
        mock_watchlist.return_value = expected

        result = get_watchlist("Default")

        assert result == expected
        mock_watchlist.assert_called_once_with(name="Default")


class TestGetCryptoPositions:
    """Tests for get_crypto_positions function."""

    @patch("robinhood_mcp.tools.rh.crypto.get_crypto_positions")
    def test_returns_positions_list(self, mock_positions: MagicMock):
        """Should return list of crypto positions."""
        expected = [
            {"currency": {"code": "BTC"}, "quantity": "0.5"},
            {"currency": {"code": "ETH"}, "quantity": "2.0"},
        ]
        mock_positions.return_value = expected

        result = get_crypto_positions()

        assert result == expected

    @patch("robinhood_mcp.tools.rh.crypto.get_crypto_positions")
    def test_returns_empty_list_on_non_list(self, mock_positions: MagicMock):
        """Should return empty list when API returns non-list."""
        mock_positions.return_value = {}

        result = get_crypto_positions()

        assert result == []


class TestGetCryptoQuote:
    """Tests for get_crypto_quote function."""

    @patch("robinhood_mcp.tools.rh.crypto.get_crypto_quote")
    def test_returns_quote(self, mock_quote: MagicMock):
        """Should return quote for crypto symbol."""
        expected = {"ask_price": "50000.00", "bid_price": "49990.00", "mark_price": "49995.00"}
        mock_quote.return_value = expected

        result = get_crypto_quote("BTC")

        assert result == expected
        mock_quote.assert_called_once_with("BTC")

    @patch("robinhood_mcp.tools.rh.crypto.get_crypto_quote")
    def test_uppercases_symbol(self, mock_quote: MagicMock):
        """Should uppercase and strip symbol."""
        mock_quote.return_value = {"mark_price": "3000.00"}

        get_crypto_quote("  eth  ")

        mock_quote.assert_called_once_with("ETH")

    def test_raises_for_empty_symbol(self):
        """Should raise RobinhoodError for empty symbol."""
        with pytest.raises(RobinhoodError) as exc_info:
            get_crypto_quote("")
        assert "non-empty string" in str(exc_info.value)

    @patch("robinhood_mcp.tools.rh.crypto.get_crypto_quote")
    def test_raises_for_no_result(self, mock_quote: MagicMock):
        """Should raise RobinhoodError when API returns non-dict."""
        mock_quote.return_value = []

        with pytest.raises(RobinhoodError) as exc_info:
            get_crypto_quote("INVALID")
        assert "No crypto quote found" in str(exc_info.value)


class TestGetCryptoHistoricals:
    """Tests for get_crypto_historicals function."""

    @patch("robinhood_mcp.tools.rh.crypto.get_crypto_historicals")
    def test_returns_historical_data(self, mock_hist: MagicMock):
        """Should return list of historical data points."""
        expected = [
            {"open_price": "49000.00", "close_price": "50000.00"},
            {"open_price": "50000.00", "close_price": "51000.00"},
        ]
        mock_hist.return_value = expected

        result = get_crypto_historicals("BTC", interval="day", span="week")

        assert result == expected
        mock_hist.assert_called_once_with("BTC", interval="day", span="week")

    def test_raises_for_invalid_interval(self):
        """Should raise RobinhoodError for invalid interval."""
        with pytest.raises(RobinhoodError) as exc_info:
            get_crypto_historicals("BTC", interval="invalid")  # type: ignore
        assert "Invalid interval" in str(exc_info.value)

    def test_raises_for_invalid_span(self):
        """Should raise RobinhoodError for invalid span."""
        with pytest.raises(RobinhoodError) as exc_info:
            get_crypto_historicals("BTC", span="invalid")  # type: ignore
        assert "Invalid span" in str(exc_info.value)

    def test_accepts_15second_interval(self):
        """Should accept 15second as a valid crypto-specific interval."""
        with patch("robinhood_mcp.tools.rh.crypto.get_crypto_historicals") as mock_hist:
            mock_hist.return_value = []
            result = get_crypto_historicals("BTC", interval="15second", span="hour")
            assert result == []


class TestSearchSymbols:
    """Tests for search_symbols function."""

    @patch("robinhood_mcp.tools.rh.stocks.get_instruments_by_symbols")
    def test_returns_search_results(self, mock_instruments: MagicMock):
        """Should return matching instruments."""
        expected = [{"symbol": "AAPL", "name": "Apple Inc."}]
        mock_instruments.return_value = expected

        result = search_symbols("AAPL")

        assert result == expected

    def test_raises_for_empty_query(self):
        """Should raise RobinhoodError for empty query."""
        with pytest.raises(RobinhoodError) as exc_info:
            search_symbols("")
        assert "non-empty string" in str(exc_info.value)
