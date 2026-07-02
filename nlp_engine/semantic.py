"""
Екстрактор ознак на основі NLP.
Модель: paraphrase-multilingual-MiniLM-L12-v2

Семантична подібність між відгуком та позитивними/негативними
прототипними реченнями для кожного критерію

"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# прототипні речення для кожного критерію
# модель обчислює косинусну подібність між відгуком і прототипними реченнями
PROTOTYPES = {
    "wait_time": {
        "positive": [
            "Прийняли швидко, без черги, майже не чекав",
            "Вчасно прийняли, мінімальне очікування",
            "Одразу потрапив до лікаря, черги не було",
            "чекала свого прийому 10 хв",
            "все було швидко",
            "швидко", "оперативно", "без затримок"
        ],
        "negative": [
            "Довго чекав у черзі, велика черга, втратив багато часу",
            "Тривале очікування, лікар запізнився на прийом",
            "Прийшлося чекати більше години, великі затримки",
            "довго чекати", "затримка прийому", "сидів під дверима"
        ],
    },
    "doctor_comp": {
        "positive": [
            "Компетентний лікар, правильно поставив діагноз, досвідчений фахівець",
            "Грамотний лікар, розібрався в проблемі, призначив правильне лікування",
            "Відмінний спеціаліст, знає свою справу, висококваліфікований","молодець",
            "Все сподобалось","гарний лікар", "професійно",
            "професіонал", "хороший лікар", "допоміг", "вирішив проблему", "спас мене", "врятував",
            "лікар дуже професійний","професійний лікар","швидко мене вилікував","лікар вилікував","допоміг вилікуватися",
            "ефективне лікування","лікування допомогло","лікар швидко допоміг","результат лікування був хороший"
        ],
        "negative": [
            "Некомпетентний лікар, неправильний діагноз, слабкий фахівець",
            "Лікар не знає що робить, помилився з діагнозом, безграмотний",
            "жахливий лікар", "не допоміг", "дарма витрачені гроші", "шарлатан"
        ],
    },
    "politeness": {
        "positive": [
            "Ввічливий та привітний персонал, уважне ставлення до пацієнтів",
            "Чуйний та доброзичливий колектив, душевна атмосфера",
            "приємний лікар", "усміхнений персонал", "гарне ставлення", "уважний"
        ],
        "negative": [
            "Грубий персонал, хамське ставлення, неввічливі співробітники",
            "Зневажливе ставлення до пацієнтів, нагрубили на рецепції",
            "хами", "грубіяни", "нахамили", "жахливе ставлення"
        ],
    },
    "cleanliness": {
        "positive": [
            "Чисто та охайно, приємна атмосфера, сучасне обладнання",
            "Затишна та чиста клініка, гарний інтер'єр, комфортно",
            "чисто", "охайно", "гарний ремонт", "стерильно"
        ],
        "negative": [
            "Брудно та неохайно, неприємний запах, старе занедбане обладнання",
            "Антисанітарія, брудна клініка, занедбане приміщення",
            "брудно", "старий ремонт", "погані умови"
        ],
    },
    "emotion": {
        "positive": [
            "Дуже задоволений, рекомендую всім, чудовий досвід, дякую","чудове враження",
            "Повернусь знову, позитивні враження, відмінна клініка","лікар найкращий",
            "супер", "рекомендую", "задоволена", "найкраща клініка", "прийду ще", "раджу","скарб","рекомендую"
        ],
        "negative": [
            "Жахливо, не рекомендую, розчарований, більше не прийду сюди",
            "Жалкую що звернувся, негативний досвід, незадоволений повністю",
            "жах", "ніколи більше", "оминайте десятою дорогою", "кошмар"
        ],
    },
    "consultation": {
        "positive": [
            "Детально все пояснив, докладна консультація, уважно вислухав пацієнта",
            "Розповів про лікування, приділив час, не поспішав, все розтлумачив",
            "Хороша, зрозуміла консультація", "гарна консультація",
            "відповів на всі питання", "розжував", "призначив лікування",
        ],
        "negative": [
            "Нічого не пояснив, поверхнева консультація, не вислухав, відписався",
            "Не приділив уваги, швидко відправив, без нормального огляду",
            "мовчав", "поспішав", "неуважний огляд"
        ],
    },
}
DOCTOR_POSITIVE = [
"гарний лікар", "професійно",
    "хороший лікар", "вирішив проблему", "спас мене", "врятував",
    "професіонал", "професійний", "професійно",
    "заради лікаря",
    "компетентний", "фахівець", "спеціаліст",
    "вилікував", "вилікувала", "допоміг", "допомогла"
]

WAIT_NEGATIVE = [
    "довго чекала", "довго чекав", "черга",
    "більше години", "затримка", "чекала прийому",
    "чекав прийому"
]
# слова та фрази, які вказують на короткий позитивний або вдячний відгук
GRATEFUL_TRIGGERS = [
    "дякую", "дякуємо", "дуже дякую", "величезне дякую", "спасибі","круто","чудовий",
    "все супер", "все ок", "все добре", "все чудово", "все сподобалось", "дуже сподобалось",
    "клас!", "клас", "рекомендую", "задоволений", "задоволена", "найкращі","раджу","скарб", "10/10", "10 з 10"
]
# відповідність окремих критеріїв до загальних категорій
DIMENSION_CATEGORIES = {
    "doctor_comp": "competence",
    "consultation": "competence",
    "wait_time": "service",
    "politeness": "service",
    "cleanliness": "service",
    "emotion": "service",
}

# score = 5 + diff * SCALE
SCALE = 18.0

# функція приводить усі фрази до нижнього регістру
def _normalize_phrases(items: list[str]) -> list[str]:
    return [item.lower().strip() for item in items]

# нормалізація всіх прототипних речень до нижнього регістру
PROTOTYPES = {
    dim: {
        tone: _normalize_phrases(phrases)
        for tone, phrases in protos.items()
    }
    for dim, protos in PROTOTYPES.items()
}

DOCTOR_POSITIVE = _normalize_phrases(DOCTOR_POSITIVE)
WAIT_NEGATIVE = _normalize_phrases(WAIT_NEGATIVE)
GRATEFUL_TRIGGERS = _normalize_phrases(GRATEFUL_TRIGGERS)


class NLPParser:

    _instance = None

    # створює або повертає вже існуючий екземпляр класу
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    # завантажує nlp-модель та обчислює ембедінги для прототипних речень
    def load(self):
        if self._ready:
            return self
        self.model = SentenceTransformer(MODEL_NAME)
        self._embeddings = {}
        for dim, protos in PROTOTYPES.items():
            self._embeddings[dim] = {
                "positive": self.model.encode(
                    protos["positive"], normalize_embeddings=True, show_progress_bar=False
                ),
                "negative": self.model.encode(
                    protos["negative"], normalize_embeddings=True, show_progress_bar=False
                ),
            }
        self._ready = True
        return self

    def _is_short_grateful(self, text: str) -> bool:
        if len(text) > 55:
            return False
        for phrase in GRATEFUL_TRIGGERS:
            if phrase in text:
                return True
        return False

    # основний метод для вилучення числових ознак із тексту відгуку
    def extract_features(self, text: str) -> dict:
        text_lower = text.lower().strip()

        if self._is_short_grateful(text_lower):
            features = {dim: 10.0 for dim in PROTOTYPES}
            found_keywords = {
                dim: {"positive": ["(вдячний короткий відгук)"], "negative": []}
                for dim in PROTOTYPES
            }
            return {
                "features": features,
                "found_keywords": found_keywords,
                "is_short_grateful": True,
                "category_highlights": {
                    "competence": ["(вдячний відгук)"],
                    "service": ["(вдячний відгук)"],
                },
            }

        # обчислення ембедінгу вхідного відгуку
        review_emb = self.model.encode(
            [text_lower], normalize_embeddings=True, show_progress_bar=False
        )[0]

        features = {}
        found_keywords = {}
        category_highlights = {"competence": [], "service": []}

        # аналіз кожного критерію окремо
        for dim, embs in self._embeddings.items():
            # середня подібність відгуку до позитивних прототипів
            sim_pos = float(np.mean(embs["positive"] @ review_emb))

            # середня подібність відгуку до негативних прототипів
            sim_neg = float(np.mean(embs["negative"] @ review_emb))

            # різниця між позитивною та негативною семантичною подібністю
            diff = sim_pos - sim_neg

            # перетворення різниці подібностей у шкалу від 0 до 10
            score = float(np.clip(5.0 + diff * SCALE, 0, 10))
            positive_hits = []
            negative_hits = []
            if dim == "doctor_comp":
                positive_hits = [w for w in DOCTOR_POSITIVE if w in text_lower]
                if positive_hits:
                    score = max(score, 7.0)

            if dim == "wait_time":
                negative_hits = [w for w in WAIT_NEGATIVE if w in text_lower]
                if not negative_hits and score < 4.5:
                    score = 5.0

            # збереження округленої оцінки за поточним критерієм
            features[dim] = round(score, 2)

            # визначння загальної категорії критерію
            category = DIMENSION_CATEGORIES[dim]
            if positive_hits:
                found_keywords[dim] = {"positive": positive_hits, "negative": []}
                category_highlights[category].extend(positive_hits)
            elif negative_hits:
                found_keywords[dim] = {"positive": [], "negative": negative_hits}
                category_highlights[category].extend(negative_hits)
            elif diff > 0.03:
                label = f"+ схожість {sim_pos:.2f}"
                found_keywords[dim] = {"positive": [label], "negative": []}
                category_highlights[category].append(label)
            elif diff < -0.03:
                label = f"- схожість {sim_neg:.2f}"
                found_keywords[dim] = {"positive": [], "negative": [label]}
                category_highlights[category].append(label)
            else:
                found_keywords[dim] = {"positive": [], "negative": []}

        # повернення всіх отриманих ознак і пояснень
        return {
            "features": features,
            "found_keywords": found_keywords,
            "is_short_grateful": False,
            "category_highlights": category_highlights,
        }


_parser = None


def extract_features(text: str) -> dict:
    global _parser
    if _parser is None:
        _parser = NLPParser().load()
    return _parser.extract_features(text)
