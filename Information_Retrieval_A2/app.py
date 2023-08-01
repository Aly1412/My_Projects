from flask import Flask, render_template, request
from Query_processor import QueryProcessor

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    return render_template('index.html')

# route to return result set
@app.route("/results", methods=["POST", "GET"])
def results():
    if request.method == 'POST':
        query = request.form['query']

        q = QueryProcessor()
        result_set = q.ProcessQuery(query)
        return render_template("result.html",tasks = result_set)

    else:
        return render_template("result.html")

# route to show result document
@app.route("/showDoc/<int:id>", methods=["POST", "GET"])
def showDoc(id):   
    try:
        return render_template(str(id) + ".txt")
    except:
        return render_template(str(id+2) + ".txt")


if __name__ == "__main__":
    app.run(debug=True)
