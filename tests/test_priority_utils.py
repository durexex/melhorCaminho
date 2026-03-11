import priority_utils as pu


def test_parse_city_priority_csv_by_index():
    cities = [(0, 0), (10, 10)]
    rules = {"critical": {}, "regular": {}}
    csv_text = "index,priority\n0,critical\n1,regular\n"
    overrides, errors = pu.parse_city_priority_csv(csv_text, cities, rules)
    assert errors == []
    assert overrides == {0: "critical", 1: "regular"}


def test_parse_city_priority_csv_by_coordinates():
    cities = [(0, 0), (10, 10)]
    rules = {"critical": {}, "regular": {}}
    csv_text = "x,y,priority\n10,10,critical\n"
    overrides, errors = pu.parse_city_priority_csv(csv_text, cities, rules)
    assert errors == []
    assert overrides == {1: "critical"}


def test_parse_city_priority_csv_invalid_priority():
    cities = [(0, 0)]
    rules = {"critical": {}}
    csv_text = "index,priority\n0,unknown\n"
    overrides, errors = pu.parse_city_priority_csv(csv_text, cities, rules)
    assert overrides == {}
    assert errors


def test_build_city_priority_csv():
    cities = [(0, 0), (5, 5)]
    overrides = {1: "critical"}
    csv_text = pu.build_city_priority_csv(cities, overrides, "regular")
    lines = [line.strip() for line in csv_text.strip().splitlines()]
    assert lines[0] == "index,x,y,priority"
    assert lines[1] == "0,0,0,regular"
    assert lines[2] == "1,5,5,critical"
