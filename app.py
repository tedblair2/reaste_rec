import Recommender
import Content
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, request
import random

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello world come here"


@app.route('/content', methods=['POST'])
def content():
    postid = request.form.get('postid')

    # content based filtering to provide recommendations
    results = Content.get_content(postid)
    history = Content.get_history()

    # using collaborative filtering to provide an extra 5 recommendations
    train_data, test_data = train_test_split(history, test_size=0.2, random_state=3)
    model = Recommender.house_recommender(5)
    model.create(train_data, 'userid', 'postid')

    similar = model.similar_items([postid])
    similar_list = list(similar['house_id'])

    results.extend(similar_list)
    results = list(dict.fromkeys(results))

    return jsonify({'postlist': results})


@app.route('/collaborative', methods=['POST'])
def collaborative():
    userid = request.form.get('userid')

    # using content based filtering to provide personalized recommendations
    user_items = Content.get_user_items(userid)

    results = []
    for item in user_items:
        recommendations = Content.get_content(item)
        results.extend(recommendations)

    results = list(dict.fromkeys(results))
    if len(results) < 12:
        results_random = results
    else:
        results_random = random.sample(results, 10)

    # using collaborative filtering to provide personalized recommendations
    history = Content.get_history()
    users = history['userid'].unique()
    houses = history['postid'].unique().tolist()
    user_list = users.tolist()
    train_data, test_data = train_test_split(history, test_size=0.2, random_state=3)
    model = Recommender.house_recommender(10)
    model.create(train_data, 'userid', 'postid')

    if userid not in user_list:
        houses_list = random.sample(houses, 10)
        return {"postlist": houses_list}
    else:
        user_results = model.recommend(userid)
        user_results_list = list(user_results['house_id'])

    user_results_list.extend(results_random)
    final_list = list(dict.fromkeys(user_results_list))

    return jsonify({'postlist': final_list})


if __name__ == "__main__":
    app.run(debug=True)
