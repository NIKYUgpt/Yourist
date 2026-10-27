from app.modules.assistant.verifier import UNKNOWN, needs_revision, parse, revision


def test_parse_valid_and_fenced_json():
    v = parse(
        '```json\n{"verdict": "partially", "issues": [{"claim": "28 дней", '
        '"problem": "не указано", "severity": "critical"}], "missing": "ч. 2"}\n```'
    )
    assert v.verdict == "partially"
    assert v.issues == [{"claim": "28 дней", "problem": "не указано", "severity": "warning"}]
    assert v.missing == "ч. 2"


def test_parse_rejects_garbage_and_unknown_verdicts():
    assert parse("всё хорошо").verdict == UNKNOWN
    assert parse('{"verdict": "ok"}').verdict == UNKNOWN
    assert parse('{"verdict": "unknown"}').verdict == UNKNOWN


def test_revision_lists_problems():
    v = parse('{"verdict": "unsupported", "issues": [{"claim": "один месяц", "problem": "три"}]}')
    assert "«один месяц»: три" in revision(v)
    assert "главный вывод не следует" in revision(parse('{"verdict": "unsupported"}'))


def test_needs_revision():
    error = '{"claim": "1 месяц", "problem": "в источнике 3", "severity": "error"}'
    warning = '{"claim": "28 дней", "problem": "не указано", "severity": "warning"}'
    assert needs_revision(parse('{"verdict": "unsupported"}'))
    assert needs_revision(parse(f'{{"verdict": "partially", "issues": [{error}]}}'))
    assert not needs_revision(parse(f'{{"verdict": "partially", "issues": [{warning}]}}'))
    assert not needs_revision(parse('{"verdict": "supported"}'))
    assert not needs_revision(parse("мусор"))
