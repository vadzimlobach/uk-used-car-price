from src.schema import RAW_FEATURES, CarTrainingFeatures


def test_raw_features_unique_and_nonempty():
    assert len(RAW_FEATURES) > 0
    assert len(set(RAW_FEATURES)) == len(RAW_FEATURES)


def test_carfeatures_validation_happy_path():
    x = CarTrainingFeatures(
        year=2019,
        mileage=45000,
        tax=150.0,
        mpg=40.0,
        engineSize=1.6,
        price=10500,
        brand="ford",
        model="fiesta",
        transmission="manual",
        fuelType="diesel",
    )
    d = x.to_dict()
    for feature in RAW_FEATURES:
        assert feature in d
