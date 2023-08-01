from asyncio import tasks
from django.http import QueryDict
from flask import Flask, render_template, request
from query_processing import Query_Processor

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    return render_template('index.html')

# route to return result set
@app.route("/results", methods=["POST", "GET"])
def results():
    if request.method == 'POST':
        query = request.form['query']

        q = Query_Processor()
        result_set = q.ProcessQuery(query)
        return render_template("result.html",tasks = result_set)

    else:
        return render_template("result.html")

# route to show result
@app.route("/View_Doc/<id>", methods=["POST", "GET"])
def showDoc(id):
    return render_template(id + ".txt")


if __name__ == "__main__":
    app.run(debug=True)
