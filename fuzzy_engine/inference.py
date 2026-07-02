"""
input: 6 критериїв (0-10), output: fuzzy rating (0-10).
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyEvaluator:
    # створює нечітку модель і калібрує шкалу рейтингу
    def __init__(self):
        self._build_model()
        self._calibrate_scale()

    # будує вхідні та вихідну змінні, терм-множини і базу правил
    def _build_model(self):

        # вхідні змінні системи, кожна має шкалу від 0 до 10
        self.wait_time = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "wait_time")
        self.doctor_comp = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "doctor_comp")
        self.politeness = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "politeness")
        self.cleanliness = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "cleanliness")
        self.emotion = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "emotion")
        self.consult_quality = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "consult_quality")

        # вихідна змінна системи — інтегральний рейтинг якості
        self.rating = ctrl.Consequent(np.arange(0, 10.1, 0.1), "rating")

        # метод дефазифікації — центр мас
        self.rating.defuzzify_method = "centroid"

        # для кожної вхідної змінної задаються три терми: low, mid, high
        for var in [
            self.wait_time,
            self.doctor_comp,
            self.politeness,
            self.cleanliness,
            self.emotion,
            self.consult_quality,
        ]:
            # низьке значення критерію
            var["low"] = fuzz.trimf(var.universe, [0, 0, 4])

            # середнє або нейтральне значення критерію
            var["mid"] = fuzz.trimf(var.universe, [3, 5, 7])

            # високе значення критерію
            var["high"] = fuzz.trimf(var.universe, [6, 10, 10])

        # терм-множина вихідної змінної rating
        self.rating["very_bad"] = fuzz.trimf(self.rating.universe, [0, 0, 3])
        self.rating["bad"] = fuzz.trimf(self.rating.universe, [2, 3.5, 5])
        self.rating["neutral"] = fuzz.trimf(self.rating.universe, [4, 6, 8])
        self.rating["good"] = fuzz.trimf(self.rating.universe, [6, 8, 9.5])
        self.rating["excellent"] = fuzz.trapmf(self.rating.universe, [8.5, 9.5, 10, 10])

        # база правил нечіткого логічного виведення
        # у системі використано 16 правил
        rules = [
            # якщо лікар компетентний і консультація якісна, рейтинг є відмінним
            ctrl.Rule(
                self.doctor_comp["high"] & self.consult_quality["high"],
                self.rating["excellent"],
            ),

            # якщо емоційне враження високе і в клініці чисто, рейтинг є відмінним
            ctrl.Rule(
                self.emotion["high"] & self.cleanliness["high"],
                self.rating["excellent"],
            ),

            # висока компетентність лікаря за середнього емоційного враження дає добрий рейтинг
            ctrl.Rule(
                self.doctor_comp["high"] & self.emotion["mid"],
                self.rating["good"],
            ),

            # висока ввічливість персоналу позитивно впливає на рейтинг
            ctrl.Rule(self.politeness["high"], self.rating["good"]),

            # короткий час очікування позитивно впливає на рейтинг
            ctrl.Rule(self.wait_time["high"], self.rating["good"]),

            # позитивне емоційне враження за середнього часу очікування дає добрий рейтинг
            ctrl.Rule(
                self.emotion["high"] & self.wait_time["mid"],
                self.rating["good"],
            ),

            # якісний лікар і консультація можуть компенсувати низьку чистоту
            ctrl.Rule(
                self.doctor_comp["high"]
                & self.consult_quality["high"]
                & self.cleanliness["low"],
                self.rating["good"],
            ),

            # висока компетентність лікаря, але низька чистота формують нейтральний рейтинг
            ctrl.Rule(
                self.doctor_comp["high"] & self.cleanliness["low"],
                self.rating["neutral"],
            ),

            # якісна консультація, але низька чистота формують нейтральний рейтинг
            ctrl.Rule(
                self.consult_quality["high"] & self.cleanliness["low"],
                self.rating["neutral"],
            ),

            # компетентний лікар, але довге очікування формують нейтральний рейтинг
            ctrl.Rule(
                self.doctor_comp["high"] & self.wait_time["low"],
                self.rating["neutral"],
            ),

            # середня компетентність лікаря відповідає нейтральному рейтингу
            ctrl.Rule(self.doctor_comp["mid"], self.rating["neutral"]),

            # середній час очікування і середнє емоційне враження дають нейтральний рейтинг
            ctrl.Rule(
                self.wait_time["mid"] & self.emotion["mid"],
                self.rating["neutral"],
            ),

            # довге очікування або неввічливість персоналу знижують рейтинг
            ctrl.Rule(
                self.wait_time["low"] | self.politeness["low"],
                self.rating["bad"],
            ),

            # низька чистота разом із негативним емоційним враженням дають поганий рейтинг
            ctrl.Rule(
                self.cleanliness["low"] & self.emotion["low"],
                self.rating["bad"],
            ),

            # низька компетентність лікаря критично знижує рейтинг
            ctrl.Rule(self.doctor_comp["low"], self.rating["very_bad"]),

            # негативне емоційне враження і погана консультація дають дуже низький рейтинг
            ctrl.Rule(
                self.emotion["low"] & self.consult_quality["low"],
                self.rating["very_bad"],
            ),
        ]

        # створення системи керування на основі бази правил
        self.rating_ctrl = ctrl.ControlSystem(rules)

        # створення симулятора нечіткого виведення
        self.rating_sim = ctrl.ControlSystemSimulation(self.rating_ctrl)

    # визначає мінімальне та максимальне сире значення рейтингу для подальшого масштабування
    def _calibrate_scale(self):

        # обчислення рейтингу для мінімальних значень усіх критеріїв
        for key in [
            "wait_time",
            "doctor_comp",
            "politeness",
            "cleanliness",
            "emotion",
            "consult_quality",
        ]:
            self.rating_sim.input[key] = 0.0

        self.rating_sim.compute()
        self.raw_min = float(self.rating_sim.output["rating"])

        # обчислення рейтингу для максимальних значень усіх критеріїв
        for key in [
            "wait_time",
            "doctor_comp",
            "politeness",
            "cleanliness",
            "emotion",
            "consult_quality",
        ]:
            self.rating_sim.input[key] = 10.0

        self.rating_sim.compute()
        self.raw_max = float(self.rating_sim.output["rating"])

    # обчислює фінальний fuzzy-рейтинг за шістьма вхідними критеріями
    def evaluate(
        self,
        wait: float,
        comp: float,
        polite: float,
        clean: float,
        emo: float,
        consult: float,
    ) -> float:

        # обмеження всіх вхідних значень діапазоном від 0 до 10
        self.rating_sim.input["wait_time"] = float(np.clip(wait, 0, 10))
        self.rating_sim.input["doctor_comp"] = float(np.clip(comp, 0, 10))
        self.rating_sim.input["politeness"] = float(np.clip(polite, 0, 10))
        self.rating_sim.input["cleanliness"] = float(np.clip(clean, 0, 10))
        self.rating_sim.input["emotion"] = float(np.clip(emo, 0, 10))
        self.rating_sim.input["consult_quality"] = float(np.clip(consult, 0, 10))

        # запуск нечіткого логічного виведення
        self.rating_sim.compute()

        # отримання сирого значення рейтингу після дефазифікації
        raw_rating = float(self.rating_sim.output["rating"])

        # якщо масштабування неможливе, повертається сирий рейтинг
        if self.raw_max <= self.raw_min:
            return raw_rating

        # лінійне масштабування рейтингу до діапазону 0-10
        scaled_rating = (
            (raw_rating - self.raw_min) / (self.raw_max - self.raw_min)
        ) * 10.0

        # фінальне обмеження рейтингу допустимим діапазоном
        final_rating = np.clip(scaled_rating, 0.0, 10.0)

        # повернення округленого результату
        return round(float(final_rating), 2)

    # обчислює рейтинг на основі словника ознак, отриманих із nlp-модуля
    def evaluate_from_features(self, features: dict) -> float:
        return self.evaluate(
            wait=features.get("wait_time", 5.0),
            comp=features.get("doctor_comp", 5.0),
            polite=features.get("politeness", 5.0),
            clean=features.get("cleanliness", 5.0),
            emo=features.get("emotion", 5.0),
            consult=features.get(
                "consultation",
                features.get("consult_quality", 5.0),
            ),
        )