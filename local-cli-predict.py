import pandas as pd
import mlflow
import click
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def predict(text):
    print(f"Accepted payload: {text}")
    my_data = {
        "selected_text" : {0: text},
        "text": {0: text},
    }
    data = pd.DataFrame(data=my_data)
    loaded_model = mlflow.pyfunc.load_model('model')
    result = loaded_model.predict(pd.DataFrame(data))
    return result

@click.command()
@click.option(
    "--tweet",
    help="Pass in text for sentiment analysis",
)
def predictcli(tweet):
    """Predict if tweet is positive, negative or neutral"""

    result = predict(tweet)
    if "negative" in result:
        click.echo(click.style(result, bg="red", fg="white"))
    elif "neutral" in result:
        click.echo(click.style(result, bg="yellow", fg="white"))
    else:
        click.echo(click.style(result, bg="green", fg="white"))

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    predictcli()