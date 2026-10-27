# Проверяющая модель: верные и испорченные ответы

Ответы модели `ollama/yandexgpt-5-lite:8b` из прогона 2026-10-27 22:14; пар (верный / с подменённым фактом): 29. Подмены — правдоподобные: другое число или противоположное утверждение в месте эталонного факта.

| Проверяющая модель | Поймала подмену | Из них «не подтверждается» | Поймала настоящие ошибки | Ложные тревоги на верных | Не разобран ответ | Медиана, с |
|---|---|---|---|---|---|---|
| `ollama/yandexgpt-5-lite:8b` | 83% | 28% | 1 из 2 | 0% | 0% | 1.5 |
| `ollama/qwen3:8b` | 93% | 41% | 1 из 2 | 14% | 0% | 3.2 |

## Подмены

- vacation-length: «28» → «42»
- probation-max: «трёх» → «четырёх»
- probation-manager: «шести» → «девяти»
- resignation-notice: «две» → «три»
- fixed-term-max: «пяти» → «семи»
- overtime-pay: «полуторном» → «одинарном»
- overtime-limit: «4» → «6»
- work-week: «40» → «60»
- teen-hours: «24» → «36»
- salary-frequency: «полмесяца» → «месяц»
- salary-delay-stop: «имеете право» → «не имеете права»
- salary-delay-compensation: «сто пятидесятой» → «трёхсотой»
- final-settlement: «в день увольнения» → «в течение месяца после увольнения»
- redundancy-pay: «среднего месячного заработка» → «двухнедельного заработка»
- redundancy-notice: «два» → «три»
- pregnant-dismissal: «не допускается» → «допускается»
- absence-definition: «четырёх» → «шести»
- discipline-deadline: «одного месяца» → «двух месяцев»
- discipline-explanation: «два» → «три»
- discipline-types: «выговор» → «штраф»
- weekend-pay: «двойном» → «полуторном»
- night-time: «22» → «23»
- court-dismissal: «одного месяца» → «двух месяцев»
- court-salary: «одного года» → «двух лет»
- min-age: «16» → «18»
- written-contract: «трёх» → «четырёх»
- maternity-leave: «70» → «105»
- childcare-leave: «трёх» → «четырёх»
- vacation-pay-timing: «три» → «четыре»

## Настоящие ошибки моделей

- court-dismissal (`ollama/qwen2.5:7b`): `ollama/yandexgpt-5-lite:8b`: partially, `ollama/qwen3:8b`: supported
- probation-manager (`yandex/aliceai-llm`): `ollama/yandexgpt-5-lite:8b`: supported, `ollama/qwen3:8b`: unsupported
