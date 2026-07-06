"""
app.py
-------
Flask web dashboard for traffic authorities to view logged violations.
Run separately from main.py:
    python app.py
Then open http://localhost:5000 in a browser.
"""

from flask import Flask, render_template, request
from database.db_manager import DBManager

app = Flask(__name__)
db = DBManager()


@app.route("/")
def dashboard():
    search_plate = request.args.get("plate", "").strip().upper()
    if search_plate:
        violations = db.fetch_by_vehicle_number(search_plate)
    else:
        violations = db.fetch_all_violations(limit=200)
    return render_template("dashboard.html", violations=violations, search_plate=search_plate)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
