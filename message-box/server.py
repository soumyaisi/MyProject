from bottle import run, post, request, response, get, route, static_file
import json
import sqlite3 as sq
QUERY = 'query'

@route('/server',method = 'POST')
def process():
    word = request.POST.get(QUERY)
    db_conn = sq.connect("test.db")
    data = (word,)
    ret_cur = db_conn.execute("SELECT swd from DATA where fwd = ? order by frequency desc;",data)
    d = ret_cur.fetchall()
    if len(d) == 0:
        return ''
    else:
        output = "<ul>"
        for t in d:
            output += "<li>" + t[0] + "</li>"
        output += "</ul>"
        return output

@route('/')
def process_front_end():
    return static_file('index.html', root=".")
# 192.168.64.35
run(host='127.0.0.1', port=7612, debug=True)