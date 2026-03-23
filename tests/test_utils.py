from datetime import datetime

from utils import ReportData, generate_report, set_report_data


def test_generate_report_includes_mutation_configuration():
    set_report_data(
        ReportData(
            num_cities=3,
            start_time=datetime(2026, 3, 23, 10, 0, 0),
            end_time=datetime(2026, 3, 23, 10, 0, 5),
            best_solution=[(0, 0), (1, 1), (2, 2)],
            best_generation=7,
            best_fitness=123.45,
            initial_mutation_probability=0.50,
            mutation_probability=0.75,
            increase_mutation_probability_if_stagnation=True,
            random_population_percent=0.90,
            initial_population_method="random",
        )
    )

    report = generate_report()

    assert "Probabilidade de mutacao inicial: 0.50" in report
    assert "Probabilidade de mutacao: 0.75" in report
    assert "Aumento de probabilidade por estagnacao: ativado" in report
