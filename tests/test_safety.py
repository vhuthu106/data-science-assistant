"""core/safety.py: guard rails for generated code (Step 7)."""
import pytest

from core.executor import new_namespace, run_code
from core.safety import SAFE_BUILTINS

LEGITIMATE = {
    "groupby": "result = df.groupby('Pclass')['Age'].mean().round(1)",
    "seaborn chart": "sns.barplot(data=df, x='Pclass', y='Fare')\nresult = 'chart'",
    "scipy t-test": "from scipy import stats\nm = df[df.Sex == 'male'].Age.dropna()\n"
                    "f = df[df.Sex == 'female'].Age.dropna()\nresult = stats.ttest_ind(m, f).pvalue",
    "helper + comprehension": "def pct(s):\n    return round(100 * s.mean(), 1)\n"
                              "result = {c: pct(df[df.Pclass == c].Survived) for c in sorted(df.Pclass.unique())}",
    "to_numeric + crosstab": "df['Fare'] = pd.to_numeric(df['Fare'])\n"
                             "result = pd.crosstab(df.Sex, df.Survived, normalize='index')",
    "import math": "import math\nresult = math.sqrt(len(df))",
}

ATTACKS = {
    "import os": "import os",
    "from subprocess import run": "from subprocess import run",
    "open a file": "open('PWNED.txt', 'w').write('x')",
    "read .env": "result = open('.env').read()",
    "__import__": "__import__('os').system('touch PWNED.txt')",
    "pandas' own os": "pd.io.common.os.system('touch PWNED.txt')",
    "matplotlib's os": "plt.matplotlib.os.listdir('.')",
    "df.to_csv": "df.to_csv('PWNED.txt')",
    "read_csv from URL": "result = pd.read_csv('http://example.com/x.csv')",
    "savefig": "plt.plot([1]); plt.savefig('PWNED.txt')",
    "np.save": "np.save('PWNED.txt', df.values)",
    "class hierarchy escape": "result = ().__class__.__bases__[0].__subclasses__()",
    "getattr trick": "result = getattr(pd, 'read_' + 'csv')('x')",
    "exec a string": "exec(\"open('PWNED.txt', 'w')\")",
}


@pytest.mark.parametrize("code", LEGITIMATE.values(), ids=LEGITIMATE.keys())
def test_legitimate_analysis_code_runs(titanic, code):
    r = run_code(code, new_namespace(titanic))
    assert r.success, r.error


@pytest.mark.parametrize("code", ATTACKS.values(), ids=ATTACKS.keys())
def test_attacks_are_blocked_before_running(titanic, code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = run_code(code, new_namespace(titanic))
    assert not r.success and r.error.startswith("Blocked for safety")
    assert not (tmp_path / "PWNED.txt").exists()


@pytest.mark.parametrize("code, error", [("f = open", NameError), ("import os", ImportError)])
def test_runtime_layer_works_on_its_own(code, error):
    # Pretend the static check missed something: the restricted builtins still stop it.
    with pytest.raises(error):
        exec(code, {"__builtins__": SAFE_BUILTINS})
