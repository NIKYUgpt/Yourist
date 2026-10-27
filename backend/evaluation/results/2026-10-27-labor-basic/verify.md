# Проверяющая модель: верные и испорченные ответы

Ответы модели `yandex/deepseek-v4-flash` из прогона 2026-10-27 22:14; пар (верный / с подменённым фактом): 29. Подмены — правдоподобные: другое число или противоположное утверждение в месте эталонного факта.

| Проверяющая модель | Поймала подмену | Из них «не подтверждается» | Поймала настоящие ошибки | Ложные тревоги на верных | Не разобран ответ | Медиана, с |
|---|---|---|---|---|---|---|
| `ollama/qwen3:8b` | 93% | 17% | 0 из 2 | 17% | 0% | 6.0 |
| `ollama/yandexgpt-5-lite:8b` | 90% | 7% | 1 из 2 | 7% | 0% | 2.2 |
| `ollama/qwen2.5:7b` | 79% | 0% | 0 из 2 | 0% | 0% | 1.6 |

## Подмены

- vacation-length: «28» → «42»
- probation-max: «три» → «четыре»
- probation-manager: «шести» → «девяти»
- resignation-notice: «две» → «три»
- fixed-term-max: «пять» → «семь»
- overtime-pay: «полуторном» → «одинарном»
- overtime-limit: «4» → «6»
- work-week: «40» → «60»
- teen-hours: «24» → «36»
- salary-frequency: «полмесяца» → «месяц»
- salary-delay-stop: «вправе» → «не вправе»
- salary-delay-compensation: «1/150» → «1/300»
- final-settlement: «в день увольнения» → «в течение месяца после увольнения»
- redundancy-pay: «среднего месячного заработка» → «двухнедельного заработка»
- redundancy-notice: «2» → «3»
- pregnant-dismissal: «не может» → «может»
- absence-definition: «4» → «6»
- discipline-deadline: «одного месяца» → «двух месяцев»
- discipline-explanation: «2» → «3»
- discipline-types: «выговор» → «штраф»
- weekend-pay: «двойном» → «полуторном»
- night-time: «22» → «23»
- court-dismissal: «1» → «2»
- court-salary: «одного года» → «двух лет»
- min-age: «16» → «18»
- written-contract: «трех» → «четырех»
- maternity-leave: «70» → «105»
- childcare-leave: «трех» → «четырех»
- vacation-pay-timing: «три» → «четыре»

## Настоящие ошибки моделей

- court-dismissal (`ollama/qwen2.5:7b`): `ollama/qwen3:8b`: supported, `ollama/yandexgpt-5-lite:8b`: partially, `ollama/qwen2.5:7b`: supported
- probation-manager (`yandex/aliceai-llm`): `ollama/qwen3:8b`: supported, `ollama/yandexgpt-5-lite:8b`: supported, `ollama/qwen2.5:7b`: supported
