from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from utils import parse_best_move
import chess
from stockfish import Stockfish
from config import sessions_path, session_path

app = Flask(__name__)
CORS(app)

stockfish = Stockfish('./stockfish/stockfish-windows-x86-64-avx2.exe')
stockfish.set_depth(20)
stockfish.set_skill_level(20)

board = chess.Board()

@app.route('/calc_move', methods=['POST'])
def post_route():
    # data = request.json
    # turn = data['turn']

    best_move = stockfish.get_best_move()
    board.push_san(best_move)
    stockfish.set_fen_position(board.fen())
    checkmate = board.is_checkmate()
    best_move = parse_best_move(best_move)
        
    
    result = {'best_move': best_move, 'checkmate' : checkmate}
    return jsonify(result)



@app.route('/checkmate', methods=['POST'])
def checkmate_route():

    board.clear()
    stockfish = Stockfish('./stockfish/stockfish-windows-x86-64-avx2.exe')

    stockfish.set_depth(20)
    stockfish.set_skill_level(20)

    print(stockfish.get_parameters())

    return jsonify({})



if __name__ == '__main__':
    app.run(debug=True)  