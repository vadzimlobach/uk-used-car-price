from src.schema import CarTrainingFeatures, get_training_features


def test_raw_features_unique_and_nonempty():
    training_features = get_training_features()
    assert len(training_features) > 0
    assert len(set(training_features)) == len(training_features)


def test_carfeatures_validation_happy_path():
    training_features = get_training_features()
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
    for feature in training_features:
        assert feature in d
