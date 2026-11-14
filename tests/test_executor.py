"""core/executor.py: running generated code (Step 5)."""
import warnings

from core.executor import new_namespace, run_code, snapshot, user_variables


def test_helper_functions_can_see_df(titanic):
    # Regression: the old exec(code, {}, locals) raised NameError inside functions.
    code = "def mean_of(col):\n    return df[col].mean()\nresult = round(mean_of('Age'), 1)"
    r = run_code(code, new_namespace(titanic))
    assert r.success and r.result == 29.7


def test_result_variable_is_the_answer(titanic):
    # Regression: the old code returned whichever DataFrame was created last.
    code = "tmp = df.head(2)\nresult = df.groupby('Pclass')['Age'].mean()\nlater = df.tail(3)"
    r = run_code(code, new_namespace(titanic))
    assert list(r.result.round(1)) == [38.2, 29.9, 25.1]


def test_figures_are_captured_as_png(titanic):
    r = run_code("df['Age'].hist()\nplt.show()", new_namespace(titanic))
    assert len(r.figures) == 1 and r.figures[0][:4] == b"\x89PNG"


def test_two_charts_give_two_images(titanic):
    code = "fig, ax = plt.subplots(); ax.plot([1, 2])\nfig2, ax2 = plt.subplots(); ax2.bar(['a'], [1])"
    assert len(run_code(code, new_namespace(titanic)).figures) == 2


def test_printed_output_is_captured(titanic):
    assert "RangeIndex: 891 entries" in run_code("df.info()", new_namespace(titanic)).output


def test_warnings_are_reported_and_not_leaked(titanic):
    # In pandas 3 this chained inplace call silently does nothing; the warning must be shown.
    before = warnings.showwarning
    code = "df['Age'].fillna(df['Age'].mean(), inplace=True)\nresult = int(df['Age'].isna().sum())"
    r = run_code(code, new_namespace(titanic))
    assert r.result == 177
    assert any("chained assignment" in w for w in r.warnings)
    assert warnings.showwarning is before


def test_error_points_at_the_failing_line(titanic):
    r = run_code("x = 1\ny = 2\nresult = df['Agee'].mean()", new_namespace(titanic))
    assert not r.success and r.error.startswith("KeyError (line 3)")


def test_syntax_error_is_reported(titanic):
    r = run_code("result = df[", new_namespace(titanic))
    assert r.error.startswith("SyntaxError (line 1)")


def test_steps_share_data_and_variables(titanic):
    ns = new_namespace(titanic)
    run_code("df['Age'] = df['Age'].fillna(df['Age'].median())\nmedian_age = df['Age'].median()", ns)
    assert run_code("result = int(df['Age'].isna().sum())", ns).result == 0
    assert "median_age" in user_variables(ns)
    assert "result" not in user_variables(ns)


def test_failed_step_on_snapshot_leaves_data_untouched(titanic):
    ns = new_namespace(titanic)
    trial = snapshot(ns)
    run_code("df['Age'] = 0\nraise ValueError('boom')", trial)
    assert ns["df"]["Age"].isna().sum() == 177


def test_no_files_are_written(titanic, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_code("df['Age'].hist()\nplt.show()", new_namespace(titanic))
    assert list(tmp_path.iterdir()) == []
