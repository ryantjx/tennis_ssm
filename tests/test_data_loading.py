"""Tests for the historical WTA workbook loader.

These tests cover the resilience behaviour added after the upstream host
stopped serving the legacy ``http://www.tennis-data.co.uk/{year}w/{year}.xlsx``
paths: link discovery from the data index page, browser-like request headers,
and local caching so a transient outage degrades rather than fails.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from src.data import data as data_module


INDEX_HTML = """
<html><body>
<a href="hrjk-85HytOjkhth76j_ygh4jf7/2025w/2025.xlsx">2025</a>
<a href='hrjk-85HytOjkhth76j_ygh4jf7/2026w/2026.xlsx'>2026</a>
<a href="hrjk-85HytOjkhth76j_ygh4jf7/2024/2024.xlsx">ATP 2024</a>
</body></html>
"""

SAMPLE_FRAME = pl.DataFrame(
    {
        "Date": ["2026-01-05"],
        "Winner": ["Player A"],
        "Loser": ["Player B"],
        "Tournament": ["Test Open"],
        "Location": ["London"],
        "Tier": ["WTA 250"],
        "Surface": ["Hard"],
        "Round": ["Final"],
    }
)


class WorkbookDiscoveryTest(unittest.TestCase):
    def test_discovers_hashed_wta_workbook_links_only(self):
        with patch.object(
            data_module, "fetch_url_bytes", return_value=INDEX_HTML.encode()
        ) as fetch:
            urls = data_module.discover_wta_workbook_urls()

        self.assertEqual(set(urls), {2025, 2026})
        self.assertTrue(urls[2026].endswith("/2026w/2026.xlsx"))
        self.assertTrue(urls[2026].startswith("https://tennis-data.co.uk/"))
        fetch.assert_called_once()

    def test_returns_empty_mapping_when_index_page_is_unavailable(self):
        with patch.object(
            data_module,
            "fetch_url_bytes",
            side_effect=RuntimeError("boom"),
        ):
            urls = data_module.discover_wta_workbook_urls()

        self.assertEqual(urls, {})

    def test_request_headers_identify_a_browser(self):
        # The host returns HTTP 403 for the default Python user agent.
        user_agent = data_module.DATA_REQUEST_HEADERS["User-Agent"]
        self.assertIn("Mozilla/5.0", user_agent)
        self.assertNotIn("Python", user_agent)


class WorkbookLoadingTest(unittest.TestCase):
    def setUp(self):
        self.cache_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        for path in self.cache_dir.glob("*"):
            path.unlink()
        self.cache_dir.rmdir()

    def test_uses_discovered_url_then_caches_workbook(self):
        with patch.object(data_module, "DATA_CACHE_DIR", self.cache_dir), patch.object(
            data_module, "fetch_url_bytes", return_value=b"workbook"
        ) as fetch, patch.object(
            data_module.pl, "read_excel", return_value=SAMPLE_FRAME
        ):
            frame = data_module.load_wta_workbook(
                2026, "https://tennis-data.co.uk/hash/2026w/2026.xlsx"
            )

        self.assertEqual(frame.height, 1)
        fetch.assert_called_once_with(
            "https://tennis-data.co.uk/hash/2026w/2026.xlsx"
        )
        self.assertEqual((self.cache_dir / "2026w.xlsx").read_bytes(), b"workbook")

    def test_falls_back_to_legacy_template_when_discovery_failed(self):
        with patch.object(data_module, "DATA_CACHE_DIR", self.cache_dir), patch.object(
            data_module, "fetch_url_bytes", return_value=b"workbook"
        ) as fetch, patch.object(
            data_module.pl, "read_excel", return_value=SAMPLE_FRAME
        ):
            data_module.load_wta_workbook(2024, None)

        fetch.assert_called_once_with(data_module.DATA_URL_TEMPLATE.format(year=2024))

    def test_tries_discovered_url_before_legacy_template(self):
        discovered = "https://tennis-data.co.uk/hash/2026w/2026.xlsx"
        with patch.object(data_module, "DATA_CACHE_DIR", self.cache_dir), patch.object(
            data_module,
            "fetch_url_bytes",
            side_effect=[RuntimeError("HTTP Error 403: Forbidden"), b"workbook"],
        ) as fetch, patch.object(
            data_module.pl, "read_excel", return_value=SAMPLE_FRAME
        ):
            data_module.load_wta_workbook(2026, discovered)

        attempted = [call.args[0] for call in fetch.call_args_list]
        self.assertEqual(
            attempted,
            [discovered, data_module.DATA_URL_TEMPLATE.format(year=2026)],
        )

    def test_uses_cached_workbook_when_download_fails(self):
        cached = self.cache_dir / "2026w.xlsx"
        cached.write_bytes(b"cached")

        with patch.object(data_module, "DATA_CACHE_DIR", self.cache_dir), patch.object(
            data_module,
            "fetch_url_bytes",
            side_effect=RuntimeError("HTTP Error 403: Forbidden"),
        ), patch.object(
            data_module, "_cache_entry_is_fresh", return_value=False
        ), patch.object(
            data_module.pl, "read_excel", return_value=SAMPLE_FRAME
        ) as read_excel:
            frame = data_module.load_wta_workbook(2026, None)

        self.assertEqual(frame.height, 1)
        read_excel.assert_called_once_with(cached)

    def test_fresh_cache_short_circuits_download(self):
        cached = self.cache_dir / "2026w.xlsx"
        cached.write_bytes(b"cached")

        with patch.object(data_module, "DATA_CACHE_DIR", self.cache_dir), patch.object(
            data_module, "fetch_url_bytes"
        ) as fetch, patch.object(
            data_module, "_cache_entry_is_fresh", return_value=True
        ), patch.object(
            data_module.pl, "read_excel", return_value=SAMPLE_FRAME
        ):
            data_module.load_wta_workbook(2026, None)

        fetch.assert_not_called()

    def test_raises_when_no_source_and_no_cache_available(self):
        with patch.object(data_module, "DATA_CACHE_DIR", self.cache_dir), patch.object(
            data_module,
            "fetch_url_bytes",
            side_effect=RuntimeError("HTTP Error 403: Forbidden"),
        ):
            with self.assertRaises(RuntimeError):
                data_module.load_wta_workbook(2026, None)

    def test_candidate_urls_deduplicate_preserving_order(self):
        duplicated = data_module.DATA_URL_TEMPLATE.format(year=2025)
        urls = data_module._workbook_url_candidates(2025, duplicated)

        self.assertEqual(urls, [duplicated])


class CacheFreshnessTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkstemp(suffix=".xlsx")[1])
        self.path.write_bytes(b"data")

    def tearDown(self):
        self.path.unlink()

    def test_completed_seasons_are_treated_as_immutable(self):
        with patch.object(data_module, "CURRENT_SEASON_CACHE_TTL_HOURS", 0.0):
            self.assertTrue(data_module._cache_entry_is_fresh(self.path, 2020))
            self.assertFalse(data_module._cache_entry_is_fresh(self.path, 2999))

    def test_empty_cache_file_is_not_fresh(self):
        self.path.write_bytes(b"")
        self.assertFalse(data_module._cache_entry_is_fresh(self.path, 2020))

    def test_missing_cache_file_is_not_fresh(self):
        self.assertFalse(
            data_module._cache_entry_is_fresh(self.path.with_suffix(".missing"), 2020)
        )


if __name__ == "__main__":
    unittest.main()
