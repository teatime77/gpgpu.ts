﻿class Mat {
    constructor() {
        // 引数のリストをArrayに変換します。
        var args = Array.prototype.slice.call(arguments);

        // 引数の最後
        var last_arg = args[args.length - 1];
        if (typeof last_arg != 'number') {
            // 引数の最後が数値でない場合

            var init = last_arg;
            args.pop();
        }

        var shape = args;

        switch (shape.length) {
            case 1:
                this.Times = 1;
                this.Depth = 1;
                this.Rows = 1;
                this.Cols = shape[0];
                break;

            case 2:
                this.Times = 1;
                this.Depth = 1;
                this.Rows = shape[0];
                this.Cols = shape[1];
                break;

            case 3:
                this.Times = 1;
                this.Depth = shape[0];
                this.Rows = shape[1];
                this.Cols = shape[2];
                break;

            case 4:
                this.Times = shape[0];
                this.Depth = shape[1];
                this.Rows  = shape[2];
                this.Cols  = shape[3];
                break;

            default:
                Assert(false, "new mat")
                break;
        }

        this.shape = shape;
        this.nElement = shape.reduce((x, y) => x * y);

        this.sizes = [];
        var n = 1;
        for (var i = 0; i < shape.length; i++) {
            
            n *= shape[shape.length - 1 - i];
            this.sizes.splice(i, 0, n);
        }

        if (init) {

            if ((init instanceof Float32Array || init instanceof Float64Array) && init.length == this.Rows * this.Cols * this.Depth) {

            }
            else {
                console.log("--------------------------------")
                console.log(init instanceof Float32Array);
                console.log(init instanceof Float64Array);
                console.log(String(init.length));
                console.log(String(this.Rows) + ":" + String(this.Cols) + ":" + String(this.Depth));
                console.log(init.length == this.Rows * this.Cols * this.Depth);

                try {
                    // 例外を発生させてみる。
                    throw new Error("original Error");
                }
                catch (e) {
                    printStackTrace(e);
                }
                console.assert(false);
            }
            Assert((init instanceof Float32Array || init instanceof Float64Array) && init.length == this.Rows * this.Cols * this.Depth, "Mat-init");
            this.dt = init;
        }
        else {

            this.dt = newFloatArray(this.Rows * this.Cols * this.Depth);
            /*
            for (var r = 0; r < this.Rows; r++) {
                for (var c = 0; c < this.Cols; c++) {
                    //                            this.dt[r * this.Cols + c] = r * 1000 + c;
                    this.dt[r * this.Cols + c] = Math.random();
                }
            }
            */
        }
    }

    copy(m) {
        Assert(this.Rows == m.Rows && this.Cols == m.Cols && this.Depth == m.Depth);
        this.dt.set(m.dt);
    }

    map(f) {
        return new Mat(this.Rows, this.Cols, this.dt.map(f));
    }

    T() {
        var v = newFloatArray(this.Cols * this.Rows);
        for (var r = 0; r < this.Cols; r++) {
            for (var c = 0; c < this.Rows; c++) {
                v[r * this.Rows + c] = this.dt[c * this.Cols + r];
            }
        }

        return new Mat(this.Cols, this.Rows, v);
    }

    transpose() {
        return this.T();
    }

    At() {
        Assert(arguments.length == this.shape.length)
        switch (this.shape.length) {
            case 2:
                Assert(arguments[0] < this.shape[0] && arguments[1] < this.shape[1], "Mat-at");
                return this.dt[arguments[0] * this.sizes[1] + arguments[1]];

            case 3:
                Assert(arguments[0] < this.shape[0] && arguments[1] < this.shape[1] && arguments[2] < this.shape[2], "Mat-at");
                return this.dt[arguments[0] * this.sizes[1] + arguments[1] * this.sizes[2] + arguments[2]];

            case 4:
                Assert(arguments[0] < this.shape[0] && arguments[1] < this.shape[1] && arguments[2] < this.shape[2] && arguments[3] < this.shape[3], "Mat-at");
                return this.dt[arguments[0] * this.sizes[1] + arguments[1] * this.sizes[2] + arguments[2] * this.sizes[3] + arguments[3]];

            default:
                Assert(false);
        }
    }

    Set() {
        // 引数のリストをArrayに変換します。
        var args = Array.prototype.slice.call(arguments);

        var val = args.pop();

        Assert(args.length == this.shape.length)
        switch (this.shape.length) {
            case 2:
                Assert(arguments[0] < this.shape[0] && arguments[1] < this.shape[1], "Mat-at");
                this.dt[arguments[0] * this.sizes[1] + arguments[1]] = val;

            case 3:
                Assert(arguments[0] < this.shape[0] && arguments[1] < this.shape[1] && arguments[2] < this.shape[2], "Mat-at");
                this.dt[arguments[0] * this.sizes[1] + arguments[1] * this.sizes[2] + arguments[2]] = val;

            case 4:
                Assert(arguments[0] < this.shape[0] && arguments[1] < this.shape[1] && arguments[2] < this.shape[2] && arguments[3] < this.shape[3], "Mat-at");
                this.dt[arguments[0] * this.sizes[1] + arguments[1] * this.sizes[2] + arguments[2] * this.sizes[3] + arguments[3]] = val;

            default:
                Assert(false);
        }
    }

    Col(c) {
        var v = newFloatArray(this.Rows);
        for (var r = 0; r < this.Rows; r++) {
            v[r] = this.dt[r * this.Cols + c];
        }

        return new Mat(this.Rows, 1, v);
    }

    Add(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == this.Cols, "Mat-add");
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] + m.dt[k];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    AddV(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == 1, "Mat-add-V");
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] + m.dt[r];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    SubV(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == 1, "Mat-sub-V");
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] - m.dt[r];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    reduce(f) {
        var v = newFloatArray(this.Rows);
        // すべての行に対し
        for (var r = 0; r < this.Rows; r++) {
            var x;
            // 列の最初の要素から順にfを適用する。
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                if (c == 0) {
                    // 最初の場合

                    x = this.dt[k];
                }
                else {
                    // 2番目以降の場合

                    x = f(x, this.dt[k]);
                }
            }
            v[r] = x;
        }

        return new Mat(this.Rows, 1, v);
    }

    Sub(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == this.Cols, "Mat-Sub");
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] - m.dt[k];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    Mul(m) {
        if (m instanceof Number) {

            return new Mat(this.Rows, this.Cols, this.dt.map(x => x * m));
        }
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == this.Cols, "Mat-Mul");
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] * m.dt[k];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    Abs() {
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = Math.abs(this.dt[k]);
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    Sum() {
        var sum = 0;
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                sum += this.dt[r * this.Cols + c];
            }
        }

        return sum;
    }

    Dot(m) {
        Assert(m instanceof Mat && m.Rows == this.Cols, "Mat-Dot");

        var v = newFloatArray(this.Rows * m.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < m.Cols; c++) {
                var sum = 0;
                for (var k = 0; k < this.Cols; k++) {
                    sum += this.dt[r * this.Cols + k] * m.dt[k * m.Cols + c];
                }
                v[r * m.Cols + c] = sum;
            }
        }
        return new Mat(this.Rows, m.Cols, v);
    }

    toString() {
        var s = "[";
        for (var r = 0; r < this.Rows; r++) {
            if (r == 0) {

                s = s + " [";
            }
            else {

                s = s + "\r\n, [";
            }

            for (var c = 0; c < this.Cols; c++) {
                if (c != 0) {

                    s = s + ", ";
                }

                s = s + this.dt[r * this.Cols + c].toFixed(7);
            }

            s = s + "]";
        }

        s = s + " ]";

        return s;
    }
}
