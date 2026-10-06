"""Validate temporal separation and corruption ground truth without model weights."""
import numpy as np
import pandas as pd

from benchmarks.retail_cpu import CONFIG, inject_prices, temporal_split, metrics


def test_invoice_spanning_temporal_boundary_is_dropped_from_both_pools():
    df = pd.DataFrame({
        'InvoiceNo': ['a', 'cross', 'cross', 'b'],
        'InvoiceDate': pd.to_datetime(['2011-07-01', '2011-07-31', '2011-08-01', '2011-09-01'])})
    train, test = temporal_split(df)
    assert train.InvoiceNo.tolist() == ['a']
    assert test.InvoiceNo.tolist() == ['b']
    assert not set(train.InvoiceNo) & set(test.InvoiceNo)


def test_injection_is_reproducible_and_changes_only_selected_prices():
    original = np.geomspace(.1, 100, 1024)
    recorded, mask, kinds = inject_prices(original, 1101)
    recorded2, mask2, kinds2 = inject_prices(original, 1101)
    np.testing.assert_array_equal(recorded, recorded2)
    np.testing.assert_array_equal(mask, mask2)
    np.testing.assert_array_equal(kinds, kinds2)
    np.testing.assert_array_equal(recorded[~mask], original[~mask])
    assert mask.sum() == round(len(original) * CONFIG['corruption_rate'])
    assert np.all(recorded[mask] != original[mask])
    assert np.all(recorded > 0)
    for kind, ratio in [('multiply_10', 10), ('divide_10', .1)]:
        np.testing.assert_allclose(recorded[kinds == kind] / original[kinds == kind], ratio)
    assert np.all(np.abs(np.log(recorded[kinds == 'price_swap'] / original[kinds == 'price_swap'])) >= np.log(2) - 1e-12)


def test_oracle_ranking_and_corrections_have_perfect_metrics():
    original = np.geomspace(.1, 100, 1024)
    recorded, mask, kinds = inject_prices(original, 1101)
    m = metrics(mask.astype(float), original.copy(), original, recorded, mask, kinds, original * .8, original * 1.2)
    assert m['precision_at_k'] == m['auroc'] == 1
    assert m['clean_80pct_interval_coverage'] == 1
    assert m['injected_error_log_mae_after'] == 0
    assert all(v == 1 for v in m['recall_at_k_by_error'].values())


def test_sample_has_train_only_categories_and_matching_splits(tmp_path, monkeypatch):
    import benchmarks.retail_cpu as retail
    dates = ['2011-07-01'] * 120 + ['2011-08-02'] * 120
    df = pd.DataFrame({
        'InvoiceNo': pd.Series([f'invoice-{i}' for i in range(240)], dtype='string'),
        'InvoiceDate': pd.to_datetime(dates),
        'StockCode': pd.Series(['seen'] * 120 + ['new'] * 120, dtype='string'),
        'Description': pd.Series([f'item {i}' for i in range(240)], dtype='string'),
        'Quantity': np.ones(240, dtype=int),
        'CustomerID': pd.Series(['c1'] * 120 + ['c2'] * 120, dtype='string'),
        'Country': pd.Series(['UK'] * 240, dtype='string'),
        'UnitPrice': np.geomspace(.1, 100, 240),
    })
    df.to_parquet(tmp_path / 'transactions.parquet', index=False)
    monkeypatch.setitem(retail.CONFIG, 'test_rows', 100)
    full = retail.sample_data(tmp_path, 100, 1101, 'full')
    no_text = retail.sample_data(tmp_path, 100, 1101, 'no_text')
    Xtr, Xte, *_ = full
    assert Xtr.StockCode.cat.categories.tolist() == ['seen']
    assert Xte.StockCode.isna().all()
    assert Xte.CustomerID.isna().all()
    assert 'UnitPrice' not in Xtr.columns
    assert 'InvoiceNo' not in Xtr.columns
    assert full[-1]['split_sha256'] == no_text[-1]['split_sha256']
    assert 'Description' not in no_text[0].columns
    assert full[-1]['invoice_groups_disjoint']
    assert full[-1]['train_rows'] == full[-1]['test_rows'] == 100
