﻿/*
    活性化関数のid
*/
var ActivationFunction = {
    none : 0,
    sigmoid: 1,
    ReLU : 2,
};

/*
    NeuralNetworkのインスタンスを作って返す。
*/
function CreateNeuralNetwork(gpgpu){
    var miniBatchSize;
    var miniBatchIdx;
    var useSoftMax = true;
    var WebGL2;
    var DataCnt;
    var L2lambda = 0.001;
    var useGradientCheck = false;
    var inGradientCheck = false;
    var Shaders = CreateNeuralNetworkShaders();
    var random = new RandomHelper();
    var net;

//  var WeightDecay = 5.0;
//  var Momentum = 0.9;

    /*
        平均の処理時間のHTML文字列を返す。

        :param Array lap_times: 処理時間の累積の配列
        :param int   cnt:       処理の回数
    */
    function meanProcessedTime(lap_times, cnt){
        return lap_times.map(x => "<td>" + (x == null ? "" : Math.round(x / cnt)) + "</td>").join("");
    }


    /*
        経過時間の計測のクラス
    */
    class Lap {
        /*        
            :param Array lap_times: 経過時間を格納する配列 
        */
        constructor(lap_times){
            this.lastTime = new Date();
            this.lapIdx = 0;
            this.lapTimes = lap_times;
        }

        /*
            経過時間を配列に追加する。
        */
        Time(){
            var prev_last_time = this.lastTime;
            this.lastTime = new Date();

            if(this.lapTimes.length <= this.lapIdx){
                this.lapTimes.push(0);
            }
            this.lapTimes[this.lapIdx] += this.lastTime - prev_last_time;
            this.lapIdx++;
        }
    }

    /*
        ニューラルネットワークのレイヤーのクラス
    */
    class Layer {
        constructor() {
        }

        /*
            初期処理

            :param Layer prev_layer: 直前のレイヤー 
        */
        init(prev_layer) {
            this.prevLayer = prev_layer;

            if (prev_layer) {
                // 直前のレイヤーがある場合

                // 直前のレイヤーの直後のレイヤーをthisにする。
                prev_layer.nextLayer = this;
            }
        }

        /*
            ミニバッチのサイズが変わった時の処理
        */
        miniBatchSizeChanged(){
            // 順伝播の処理時間の配列
            this.forwardTime = [];

            // 誤差逆伝播の処理時間の配列
            this.backwardTime = [];

            // パラメータの更新の処理時間の配列
            this.updateTime = [];
        }

        /*
            順伝播
        */
        forward() {
        }

        /*
            誤差逆伝播
        */
        backpropagation() {
        }

        /*
            パラメータの更新
        */
        updateParameter() {
        }

        /*
            WebGLのリソースのクリア
        */
        clear(){
            if(this.params){

                for(var key in this.params){
                    WebGL2.clear(this.params[key].id);
                }
                this.params = {};
            }
        }

        /*
            勾配の計算のチェック

            :param float[] batch_Y: 正解の出力
            :param float[] exp_work: 作業用データ
            :param double  cost: コスト
            :param int     batch_idx: ミニバッチ内のインデックス
            :param int     layer_idx: レイヤーのインデックス
        */
        gradientCheck(batch_Y, exp_work, cost, batch_idx, layer_idx){
            if(! this.prevLayer){
                return;
            }

            var last_layer = net.layers[net.layers.length - 1];
            var last_delta_y_dt = new Float32Array(last_layer.deltaY.dt.length);

            if(this.deltaX){

                var prev_layer = this.prevLayer;

                var idx_list = random.RandomSampling(prev_layer.unitSize, 3);

                for(let i of idx_list){
                    var k = batch_idx * prev_layer.unitSize + i;
                    var x = prev_layer.y_.dt[k];
                    var dx = this.deltaX.dt[k];
                    var eps = x * 0.01;

                    prev_layer.y_.dt[k] = x - eps;
                    var cost1 = net.forwardCost(batch_Y, exp_work, batch_idx, layer_idx, last_delta_y_dt);

                    prev_layer.y_.dt[k] = x + eps;
                    var cost2 = net.forwardCost(batch_Y, exp_work, batch_idx, layer_idx, last_delta_y_dt);

                    var diff = dx * 2 * eps - (cost2 - cost1);
                    console.log("delta-X : %f dC:%f eps:%f cost:%f,%f,%f", diff, dx, eps, cost1, cost - cost1, cost2 - cost);

                    prev_layer.y_.dt[k] = x;
                }
            }

            if(this.deltaBias){

                net.paramGradientCheck("bias", this.bias.dt, this.deltaBias.dt, batch_Y, exp_work, cost, batch_idx, layer_idx, last_delta_y_dt);
            }

            if(this.deltaWeight){

                net.paramGradientCheck("weight", this.weight.dt, this.deltaWeight.dt, batch_Y, exp_work, cost, batch_idx, layer_idx, last_delta_y_dt);
            }
        }

        /*
            処理時間の計測値のHTML文字列を返す。

            :param int cnt: 処理の回数
        */
        processedTime(cnt){
            return "<tr><td></td>" + meanProcessedTime([], cnt) + "</tr>";
        }
    }

    /*
        入力層のクラス
    */
    class InputLayer extends Layer {

        /*
            :param int channel_size: チャネル数
            :param int rows: 行数
            :param int cols: 列数
        */
        constructor(channel_size, rows, cols) {
            super();

            this.numChannels = channel_size;
            this.numRows = rows;
            this.numCols = cols;
            this.unitSize = rows * cols;
        }
    }

    /*
        アフィン変換層のクラス
        全結合層と畳み込み層のスーパークラス
    */
    class AffineTransformationLayer extends Layer{
        /*
            CPUによるδzの計算
        */
        cpuDeltaZ(){

            // 活性化関数
            switch(this.activationFunction){
            case ActivationFunction.none:
                for (var i = 0; i < this.deltaZ.dt.length; i++) {
                    this.deltaZ.dt[i] = this.deltaY.dt[i];
                }
                break;

            case ActivationFunction.sigmoid:
                for (var i = 0; i < this.deltaZ.dt.length; i++) {
                    this.deltaZ.dt[i] = this.deltaY.dt[i] * sigmoid_prime(this.z_.dt[i]);
                }
                break;

            case ActivationFunction.ReLU:
                for (var i = 0; i < this.deltaZ.dt.length; i++) {
                    this.deltaZ.dt[i] = (this.z_.dt[i] <= 0 ? 0 : this.deltaY.dt[i]);
                }
                break;
            }
        }
    }

    /*
        全結合層のクラス
    */
    class FullyConnectedLayer extends AffineTransformationLayer {
        /*
            :param int size: 出力のニューロンの数
            :param int activation_function: 活性化関数のid
        */
        constructor(size, activation_function) {
            super();

            this.unitSize = size;
            this.activationFunction = activation_function;
            this.params = {};
        }

        /*
            初期処理

            :param Layer prev_layer: 直前のレイヤー 
        */
        init(prev_layer) {
            super.init(prev_layer);

            this.bias = random.randn(this.unitSize, 1);
            this.weight = random.randn(this.unitSize, this.prevLayer.unitSize);

            if(this.activationFunction == ActivationFunction.ReLU){
                var sd = Math.sqrt(2.0 / prev_layer.unitSize);
                for(var i = 0; i < this.weight.dt.length; i++){
                    this.weight.dt[i] *= sd;
                }
            }

            this.deltaWeight = new ArrayView(this.unitSize, this.prevLayer.unitSize);
        }

        /*
            ミニバッチのサイズが変わった時の処理
        */
        miniBatchSizeChanged(){
            super.miniBatchSizeChanged();

            this.outZero    = new Float32Array(miniBatchSize * this.unitSize);
            this.z_          = new ArrayView(miniBatchSize,  this.unitSize);
            this.y_ = new ArrayView(miniBatchSize,  this.unitSize);
            this.deltaX     = new ArrayView(miniBatchSize,  this.prevLayer.unitSize);

            this.deltaZ     = new ArrayView(miniBatchSize,  this.unitSize);

            if(!this.nextLayer){
                // 最後の場合

                this.deltaY = new ArrayView(miniBatchSize,  this.unitSize);
            }
        }

        /*
            GPUによる順伝播

            .. math::


                z_{i} = \displaystyle \sum_{j }^{ X } x_{j} \cdot weight_{i,j} + bias_{i}

                y_{i} = σ(z_{i})
        */
        gpuForward(){
            var vertex_shader = Shaders.FullyConnectedLayer_Forward;

            this.param = {
                id : "Fully-Connected-Layer-forward," + miniBatchSize + "," + this.prevLayer.unitSize + "," + this.unitSize,
                vertexShader: vertex_shader,
                args : {
                    "activationFunction": this.activationFunction,
                    "zero": this.outZero,
                    "X": WebGL2.makeTextureInfo("float", [ miniBatchSize, this.prevLayer.unitSize], this.prevLayer.y_.dt),
                    "W": WebGL2.makeTextureInfo("float", this.weight.shape, this.weight.dt),
                    "Bias": WebGL2.makeTextureInfo("float", [ 1, this.bias.dt.length ], this.bias.dt),
                    "z": this.z_.dt,
                    "y" : this.y_.dt
                }
            };

            WebGL2.compute(this.param);
        }

        /*
            順伝播
        */
        forward() {
            var lap = new Lap(this.forwardTime);
            this.gpuForward();

            lap.Time();
        }

        /*
            GPUによるδxの計算
        */
        gpuDeltaX(){
            var vertex_shader = Shaders.FullyConnectedLayer_DeltaX;

            var param_id = "Fully-Connected-Layer-gpu-delta-X," + miniBatchSize + "," + this.prevLayer.unitSize + "," + this.unitSize;
            if (this.params[param_id] == undefined){

                this.params[param_id] = {
                    id : param_id,
                    vertexShader: vertex_shader,
                    args : {
                        "zero": new Float32Array(miniBatchSize * this.prevLayer.unitSize),
                        "W": makeTextureInfo(WebGL2, "float", this.weight),
                        "deltaZ": makeTextureInfo(WebGL2, "float", this.deltaZ),
                        "deltaX" : this.deltaX.dt
                    }
                };
            }

            var param = this.params[param_id];
            param.args["deltaZ"].value = this.deltaZ.dt;

            WebGL2.compute(param);
        }

        /*
            CPUによるδxの計算

            .. math::

                \delta x_{j} = \displaystyle \sum_i^y \delta z_i \cdot weight_{i,j}
        */
        cpuDeltaX(){

            // 出力先
            var output_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // 入力に対し
                for (var x_idx = 0; x_idx < this.prevLayer.unitSize; x_idx++) {

                    var sum = 0.0;

                    // 重みの行とδzの内積
                    for (var k = 0; k < this.weight.nrow; k++) {
                        var weight_idx = k * this.weight.ncol + x_idx;
                        var delta_z_idx = batch_idx * this.unitSize + k;
                        sum += this.deltaZ.dt[delta_z_idx] * this.weight.dt[weight_idx];
                    }

                    this.deltaX.dt[output_idx] = sum;
                    output_idx++;
                }
            }
        }

        /*
            GPUによるδweightの計算

            .. math::

                \delta weight_{i,j} = \delta z_{i} \cdot x_{j}
        */
        gpuDeltaWeight(){
            var vertex_shader = Shaders.FullyConnectedLayer_DeltaWeight;

            var prev_layer = this.prevLayer;

            var vertex_shader = vertex_shader
                .replace(/miniBatchSize/g, miniBatchSize.toString())
                .replace(/WeightColSize/g, prev_layer.unitSize.toString());

            var param_id = "Fully-Connected-Layer-delta-weight," + miniBatchSize + "," + prev_layer.unitSize + "," + this.unitSize;
            if (this.params[param_id] == undefined){

                this.params[param_id] = {
                    id : param_id,
                    vertexShader: vertex_shader,
                    args : {
                        "zero": new Float32Array(this.deltaWeight.dt.length),
                        "prev_y": makeTextureInfo(WebGL2, "float", new ArrayView(miniBatchSize, prev_layer.unitSize)),
                        "deltaZ": makeTextureInfo(WebGL2, "float", this.deltaZ),
                        "deltaWeight" : this.deltaWeight.dt
                    }
                };
            }

            var param = this.params[param_id];

            param.args["prev_y"].value = prev_layer.y_.dt;;
            param.args["deltaZ"].value = this.deltaZ.dt;
            param.args["deltaWeight"].value = this.deltaWeight.dt;

            WebGL2.compute(param);
        }

        /*
            誤差逆伝播
        */
        backpropagation() {
            var lap = new Lap(this.backwardTime);

            if (this.nextLayer) {
                // 最後のレイヤーでない場合

                this.deltaY = this.nextLayer.deltaX;
            }

            this.cpuDeltaZ();

            lap.Time();

            this.deltaBias = this.deltaZ.Reduce((x, y) => x + y);
            lap.Time();

            this.gpuDeltaWeight();
            lap.Time();

            if(! (this.prevLayer instanceof InputLayer)){

                this.gpuDeltaX();
                if(Math_random() < 0.01){

                    var gpu_delta_x = new Float32Array(this.deltaX.dt);
                    this.cpuDeltaX();

                    var diff = this.deltaX.diff(gpu_delta_x);
//                    Assert(diff < 0.01, "delta-X");
                    if(0.01 < diff){
                        console.log("dense delta-X %f", diff)
                    }
                }
            }
            lap.Time();
        }

        /*
            パラメータの更新
        */
        updateParameter() {
            var lap = new Lap(this.updateTime);
            var eta = net.learningRate / miniBatchSize;

//          var c = 1.0 - net.learningRate * WeightDecay / DataCnt;

            for(var i = 0; i < this.weight.dt.length; i++){
                this.weight.dt[i] -= (eta * this.deltaWeight.dt[i]  + net.learningRate * L2lambda * this.weight.dt[i]);
    /*
                var v = Momentum * this.weightV.dt[i] - eta * this.deltaWeight.dt[i];
                this.weightV.dt[i] = v;

                this.weight.dt[i] = c * this.weight.dt[i] + v;
    */
            }
            lap.Time();

            for(var i = 0; i < this.bias.dt.length; i++){
                this.bias.dt[i] -= eta * this.deltaBias.dt[i];
            }
            lap.Time();
        }

        /*
            処理時間の計測値のHTML文字列を返す。

            :param int cnt: 処理の回数
        */
        processedTime(cnt){
            return "<tr><td>全結合層</td>" + meanProcessedTime([ this.forwardTime[0] ].concat(this.backwardTime).concat(this.updateTime[0]), cnt) + "</tr>";
        }
    }

    /*
        畳み込み層のクラス
    */
    class ConvolutionalLayer extends AffineTransformationLayer {
        
        /*        
            :param int filter_size: フィルターのサイズ
            :param int channel_size: チャネル数
            :param int activation_function: 活性化関数のid
        */
        constructor(filter_size, channel_size, activation_function) {
            super();

            this.filterSize = filter_size;
            this.numChannels = channel_size;
            this.activationFunction = activation_function;

            this.params = {};
        }

        /*
            初期処理
        
            :param Layer prev_layer: 直前のレイヤー 
        */
        init(prev_layer) {
            super.init(prev_layer);

            this.numRows = this.prevLayer.numRows - this.filterSize + 1;
            this.numCols = this.prevLayer.numCols - this.filterSize + 1;
            this.unitSize = this.numChannels * this.numRows * this.numCols;

            this.bias = new ArrayView(this.numChannels);
            for (var i = 0; i < this.bias.dt.length; i++) {
                this.bias.dt[i] = random.randn();
            }

            this.weight = new ArrayView(this.numChannels, prev_layer.numChannels, this.filterSize, this.filterSize);
            for (var i = 0; i < this.weight.dt.length; i++) {
                this.weight.dt[i] = random.randn();
            }

            if(this.activationFunction == ActivationFunction.ReLU){
                var sd = Math.sqrt(2.0 / prev_layer.unitSize);
                for(var i = 0; i < this.weight.dt.length; i++){
                    this.weight.dt[i] *= sd;
                }
            }

            this.deltaBias    = new ArrayView(this.numChannels);
            this.deltaWeight   = new ArrayView(this.numChannels, prev_layer.numChannels, this.filterSize, this.filterSize);
        }

        /*
            ミニバッチのサイズが変わった時の処理
        */
        miniBatchSizeChanged(){
            super.miniBatchSizeChanged();

            this.z_ = new ArrayView(miniBatchSize, this.unitSize);
            this.y_ = new ArrayView(miniBatchSize, this.unitSize);
            this.zero = new Float32Array(miniBatchSize * this.unitSize);

            this.deltaZ     = new ArrayView(miniBatchSize,  this.unitSize);

            if(this.prevLayer instanceof InputLayer){

                this.deltaX = undefined;
            }
            else{

                this.deltaX = new ArrayView(miniBatchSize, this.prevLayer.unitSize);
            }

        }

        /*
            GPUによる順伝播
        */
        gpuForward() {
            var prev_layer = this.prevLayer;

            var param_id = "ConvolutionalLayer-forward:" + this.filterSize + ":" + prev_layer.numChannels + ":" + this.numChannels + ":" + this.numRows + ":" + this.numCols + ":" + miniBatchSize;

            if (this.params[param_id] == undefined) {

                var shader_src = Shaders.ConvolutionalLayer_Forward
                    .replace(/numChannels/g, this.numChannels.toString() + "u")
                    .replace(/prevNumChannels/g, prev_layer.numChannels.toString() + "u")
                    .replace(/numRows/g, this.numRows.toString() + "u")
                    .replace(/numCols/g, this.numCols.toString() + "u")
                    .replace(/filterSize/g, this.filterSize.toString() + "u");

                this.params[param_id]  = {
                    id : param_id,
                    vertexShader: shader_src,
                    args : {
                        "activationFunction": this.activationFunction,
                        "zero": this.zero,
                        "prev_y": makeTextureInfo(WebGL2, "float", new ArrayView(miniBatchSize * prev_layer.numChannels, prev_layer.numRows, prev_layer.numCols)),
                        "weight": makeTextureInfo(WebGL2, "float", new ArrayView(this.numChannels * prev_layer.numChannels, this.filterSize, this.filterSize)),
                        "bias": this.bias.dt,
                        "z": this.z_.dt,
                        "y": this.y_.dt
                    }
                };
            }

            var param = this.params[param_id];

            param.args["prev_y"].value = prev_layer.y_.dt;
            param.args["weight"].value = this.weight.dt;
            WebGL2.compute(param);
        }

        /*
            CPUによる順伝播
        */
        cpuForward() {
            var prev_layer = this.prevLayer;

            var prev_y_dt = prev_layer.y_.dt;
            var z_dt = this.z_.dt;
            var y_dt = this.y_.dt;

            // 出力先
            var output_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // すべての特徴マップに対し
                for (var channel_idx = 0; channel_idx < this.numChannels; channel_idx++) {

                    // 出力の行に対し
                    for (var r1 = 0; r1 < this.numRows; r1++) {

                        // 出力の列に対し
                        for (var c1 = 0; c1 < this.numCols; c1++) {

                            var sum = 0.0;
                            var weight_idx = channel_idx * prev_layer.numChannels * this.filterSize * this.filterSize;
                            var prev_y_base = batch_idx * prev_layer.numChannels * prev_layer.numRows * prev_layer.numCols;

                            // 入力のチャネルに対し
                            for(var prev_channel_idx = 0; prev_channel_idx < prev_layer.numChannels; prev_channel_idx++){

                                // フィルターの行に対し
                                for (var r2 = 0; r2 < this.filterSize; r2++) {

                                    // フィルターの列に対し
                                    for (var c2 = 0; c2 < this.filterSize; c2++) {
                                        var prev_y_idx = prev_y_base + (r1 + r2) * prev_layer.numCols + (c1 + c2);
                                        sum += prev_y_dt[prev_y_idx] * this.weight.dt[weight_idx];
                                        weight_idx++;
                                    }
                                }
                                prev_y_base += prev_layer.numRows * prev_layer.numCols;
                            }

                            var z_val = sum + this.bias.dt[channel_idx];

                            z_dt[output_idx] = z_val;

                            // 活性化関数
                            switch(this.activationFunction){
                            case ActivationFunction.none:
                                y_dt[output_idx] = z_val;
                                break;

                            case ActivationFunction.sigmoid:
                                y_dt[output_idx] =  sigmoid(z_val);
                                break;

                            case ActivationFunction.ReLU:
                                y_dt[output_idx] = (0.0 < z_val ? z_val : 0.0);
                                break;
                            }

                            output_idx++;
                        }

                    }
                }
            }
        }

        /*
            順伝播
        */
        forward() {
            var lap = new Lap(this.forwardTime);

            this.gpuForward();

            lap.Time();

            if(miniBatchIdx == 0 || Math_random() < 0.01){

                var z_gpu_dt          = new Float32Array(this.z_.dt);
                var y_gpu_dt = new Float32Array(this.y_.dt);

                this.cpuForward();

                var max_diff = 0;

                // 出力先
                var output_idx = 0;
                for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {
                    for (var channel_idx = 0; channel_idx < this.numChannels; channel_idx++) {
                        for (var r1 = 0; r1 < this.numRows; r1++) {
                            for (var c1 = 0; c1 < this.numCols; c1++) {
                                var diff = Math.max(Math.abs(z_gpu_dt[output_idx] - this.z_.dt[output_idx]), Math.abs(y_gpu_dt[output_idx] - this.y_.dt[output_idx]));
                                if (max_diff < diff) {
                                    if(0.001 < diff){
                                        console.log("CNN forward : %dx%d %d %d %d %d %f", this.numRows, this.numCols, batch_idx, channel_idx, r1, c1, diff)
                                    }
                                    max_diff = diff;
                                }
                                output_idx++;
                            }
                        }
                    }
                }
            }
            lap.Time();
        }

        /*
            GPUによるδweightの計算
        */
        gpuDeltaWeight() {
            var prev_layer = this.prevLayer;

            var param_id = "ConvolutionalLayer-dabla-weight:" + this.filterSize + ":" + prev_layer.numChannels + ":" + this.numChannels + ":" + this.numRows + ":" + this.numCols + ":" + miniBatchSize;

            if (this.params[param_id] == undefined) {

                var shader_src = Shaders.ConvolutionalLayer_DeltaWeights
                    .replace(/miniBatchSize/g, miniBatchSize.toString() + "u")
                    .replace(/numChannels/g, this.numChannels.toString() + "u")
                    .replace(/prevNumChannels/g, prev_layer.numChannels.toString() + "u")
                    .replace(/numRows/g, this.numRows.toString() + "u")
                    .replace(/numCols/g, this.numCols.toString() + "u")
                    .replace(/filterSize/g, this.filterSize.toString() + "u");

                this.params[param_id]  = {
                    id : param_id,
                    vertexShader: shader_src,
                    args : {
                        "zero": new Float32Array(this.weight.dt.length),
                        "prev_y": makeTextureInfo(WebGL2, "float", new ArrayView(miniBatchSize * prev_layer.numChannels, prev_layer.numRows, prev_layer.numCols)),
                        "delta_z": makeTextureInfo(WebGL2, "float", new ArrayView(miniBatchSize * this.numChannels, this.numRows, this.numCols)),
                        "delta_w": this.deltaWeight.dt
                    }
                };
            }

            var param = this.params[param_id];

            param.args["prev_y"].value = prev_layer.y_.dt;
            param.args["delta_z"].value = this.deltaZ.dt;
            WebGL2.compute(param);
        }

        /*
            GPUによるδxの計算
        */
        gpuDeltaX() {
            var prev_layer = this.prevLayer;

            var param_id = "ConvolutionalLayer-delta-X:" + this.filterSize + ":" + prev_layer.numChannels + ":" + this.numChannels + ":" + this.numRows + ":" + this.numCols + ":" + miniBatchSize;

            if (this.params[param_id] == undefined) {

                var shader_src = Shaders.ConvolutionalLayer_DeltaX
                    .replace(/miniBatchSize/g, miniBatchSize.toString() + "u")
                    .replace(/numChannels/g, this.numChannels.toString() + "u")
                    .replace(/prevNumChannels/g, prev_layer.numChannels.toString() + "u")
                    .replace(/prevNumRows/g, prev_layer.numRows.toString() + "u")
                    .replace(/prevNumCols/g, prev_layer.numCols.toString() + "u")
                    .replace(/numRows/g, this.numRows.toString() + "u")
                    .replace(/numCols/g, this.numCols.toString() + "u")
                    .replace(/filterSize/g, this.filterSize.toString() + "u");

                this.params[param_id]  = {
                    id : param_id,
                    vertexShader: shader_src,
                    args : {
                        "zero": new Float32Array(this.deltaX.dt.length),
                        "weight": makeTextureInfo(WebGL2, "float", new ArrayView(this.numChannels * prev_layer.numChannels, this.filterSize, this.filterSize)),
                        "delta_z": makeTextureInfo(WebGL2, "float", new ArrayView(miniBatchSize * this.numChannels, this.numRows, this.numCols)),
                        "delta_x": this.deltaX.dt
                    }
                };
            }

            var param = this.params[param_id];

            param.args["weight"].value = this.weight.dt;
            param.args["delta_z"].value = this.deltaZ.dt;
            WebGL2.compute(param);
        }

        /*
            CPUによるδweightの計算
        */
        cpuDeltaWeight() {
            var prev_layer = this.prevLayer;
            var num_rows_cols = this.numRows * this.numCols;
            var prev_num_rows_cols = prev_layer.numRows * prev_layer.numCols;

            // 出力のチャネルに対し
            var weight_idx = 0;
            for (var channel_idx = 0; channel_idx < this.numChannels; channel_idx++) {

                // 入力のチャネルに対し
                for(var prev_channel_idx = 0; prev_channel_idx < prev_layer.numChannels; prev_channel_idx++){

                    // フィルターの行に対し
                    for (var r2 = 0; r2 < this.filterSize; r2++) {

                        // フィルターの列に対し
                        for (var c2 = 0; c2 < this.filterSize; c2++) {

                            var delta_w = 0.0;

                            // 出力の行に対し
                            for (var r1 = 0; r1 < this.numRows; r1++) {

                                // 出力の列に対し
                                for (var c1 = 0; c1 < this.numCols; c1++) {

                                    var delta_z_idx = channel_idx * num_rows_cols + r1 * (this.numCols | 0) + c1;
                                    var prev_y_idx = prev_channel_idx * prev_num_rows_cols + (r1 + r2) * (prev_layer.numCols | 0) + (c1 + c2);

                                    // バッチ内のデータに対し
                                    for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                                        var delta = this.deltaZ.dt[delta_z_idx];
                                        if (delta != 0) {

                                            delta_w += delta * prev_layer.y_.dt[prev_y_idx];
                                        }

                                        delta_z_idx += this.unitSize;
                                        prev_y_idx += (prev_layer.unitSize | 0);
                                    }
                                }
                            }

                            this.deltaWeight.dt[weight_idx] = delta_w;
                            weight_idx++;
                        }
                    }
                }
            }
            Assert(weight_idx == this.deltaWeight.dt.length);
        }

        /*
            CPUによるδbiasの計算
        */
        cpuDeltaBias(){
            var num_rows_cols = this.numRows * this.numCols;

            // すべての特徴マップに対し
            for (var channel_idx = 0; channel_idx < this.numChannels; channel_idx++) {

                var delta_bias = 0.0;

                // 出力の行に対し
                for (var r1 = 0; r1 < this.numRows; r1++) {

                    // 出力の列に対し
                    for (var c1 = 0; c1 < this.numCols; c1++) {

                        // バッチ内のデータに対し
                        var delta_z_idx = channel_idx * num_rows_cols + r1 * (this.numCols | 0) + c1;
                        for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                            delta_bias += this.deltaZ.dt[delta_z_idx];
                            delta_z_idx += this.unitSize;
                        }
                    }
                }

                this.deltaBias.dt[channel_idx] = delta_bias / (this.numRows * this.numCols);
            }
        }

        /*
            CPUによるδxの計算
        */
        cpuDeltaX2() {
            var prev_layer = this.prevLayer;
            var delta_x = new Float32Array(miniBatchSize * prev_layer.unitSize);

            var prev_y_dt = prev_layer.y_.dt;
            var z_dt = this.z_.dt;

            // 出力先
            var output_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // すべての特徴マップに対し
                for (var channel_idx = 0; channel_idx < this.numChannels; channel_idx++) {

                    // 出力の行に対し
                    for (var r1 = 0; r1 < this.numRows; r1++) {

                        // 出力の列に対し
                        for (var c1 = 0; c1 < this.numCols; c1++) {

                            var sum = 0.0;
                            var weight_idx = channel_idx * prev_layer.numChannels * this.filterSize * this.filterSize;
                            var prev_y_base = batch_idx * prev_layer.numChannels * prev_layer.numRows * prev_layer.numCols;

                            // 入力のチャネルに対し
                            for(var prev_channel_idx = 0; prev_channel_idx < prev_layer.numChannels; prev_channel_idx++){

                                // フィルターの行に対し
                                for (var r2 = 0; r2 < this.filterSize; r2++) {

                                    // フィルターの列に対し
                                    for (var c2 = 0; c2 < this.filterSize; c2++) {
                                        var prev_y_idx = prev_y_base + (r1 + r2) * prev_layer.numCols + (c1 + c2);
                                        sum += prev_y_dt[prev_y_idx] * this.weight.dt[weight_idx];

                                        delta_x[prev_y_idx] += this.deltaZ.dt[output_idx] * this.weight.dt[weight_idx]
                                        weight_idx++;
                                    }
                                }
                                prev_y_base += prev_layer.numRows * prev_layer.numCols;
                            }

                            output_idx++;
                        }
                    }
                }
            }

            return delta_x;
        }

        /*
            CPUによるδxの計算
        */
        cpuDeltaX() {
            var prev_layer = this.prevLayer;
            var num_rows_cols = this.numRows * this.numCols;

            var prev_y_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // 入力のチャネルに対し
                for(var prev_channel_idx = 0; prev_channel_idx < prev_layer.numChannels; prev_channel_idx++){

                    // 入力の行に対し
                    for (var r3 = 0; r3 < prev_layer.numRows; r3++) {

                        // 入力の列に対し
                        for (var c3 = 0; c3 < prev_layer.numCols; c3++) {

                            var sum = 0.0;

                            // 出力のチャネルに対し
                            for(var channel_idx = 0; channel_idx < this.numChannels; channel_idx++){
                                var delta_z_base = batch_idx * this.unitSize + channel_idx * num_rows_cols;
                                var weight_base = (channel_idx * prev_layer.numChannels + prev_channel_idx) * this.filterSize * this.filterSize;

                                // フィルターの行に対し
                                for (var r2 = 0; r2 < this.filterSize; r2++) {

                                    // 出力の行
                                    var r1 = r3 - r2;

                                    if(0 <= r1 && r1 < this.numRows){

                                        // フィルターの列に対し
                                        for (var c2 = 0; c2 < this.filterSize; c2++) {

                                            // 出力の列
                                            var c1 = c3 - c2;

                                            if(0 <= c1 && c1 < this.numCols){

                                                var delta_z_idx = delta_z_base + r1 * this.numCols + c1;
                                                var weight_idx = weight_base + r2 * this.filterSize + c2;
                                                sum += this.deltaZ.dt[delta_z_idx] * this.weight.dt[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }

                            this.deltaX.dt[prev_y_idx] = sum;
                            prev_y_idx++;
                        }
                    }
                }
            }
            Assert(prev_y_idx == miniBatchSize * prev_layer.unitSize);
        }

        /*
            誤差逆伝播
        */
        backpropagation() {
            var lap = new Lap(this.backwardTime);

            if (this.nextLayer) {
                // 最後のレイヤーでない場合

                this.deltaY = this.nextLayer.deltaX;
            }

            this.cpuDeltaZ();
            lap.Time();

            this.cpuDeltaBias();
            lap.Time();

            this.gpuDeltaWeight();

            if(miniBatchIdx == 0 || Math_random() < 0.01){

                var delta_w = new Float32Array(this.deltaWeight.dt);
                this.cpuDeltaWeight();

                var diff = this.deltaWeight.diff(delta_w);
                if(0.00001 < diff){
                    console.log("CNN delta-W diff:%f", diff);
                }
            }

            lap.Time();

            if(! (this.prevLayer instanceof InputLayer)){
                // 直前が入力層でない場合

                this.gpuDeltaX();

                if(miniBatchIdx == 0 || Math_random() < 0.01){

                    var delta_x1 = new Float32Array(this.deltaX.dt);

                    this.cpuDeltaX();

                    var delta_x2 = this.cpuDeltaX2();
                    var diff1 = this.deltaX.diff(delta_x1);
                    var diff2 = this.deltaX.diff(delta_x2);
                    if(0.00001 < diff1 || 0.00001 < diff2){
                        console.log("CNN delta-X diff:%f %f", diff1, diff2);
                    }
                }
            }
            lap.Time();
        }

        /*
            パラメータの更新
        */
        updateParameter() {
            var lap = new Lap(this.updateTime);

            var prev_layer = this.prevLayer;
            var eta = net.learningRate / miniBatchSize;

            var weight_idx = 0;

            // 出力のチャネルに対し
            for (var channel_idx = 0; channel_idx < this.numChannels; channel_idx++) {

                this.bias.dt[channel_idx] -= eta * this.deltaBias.dt[channel_idx];

                // 入力のチャネルに対し
                for(var prev_channel_idx = 0; prev_channel_idx < prev_layer.numChannels; prev_channel_idx++){

                    // フィルターの行に対し
                    for (var r2 = 0; r2 < this.filterSize; r2++) {

                        // フィルターの列に対し
                        for (var c2 = 0; c2 < this.filterSize; c2++) {
                            this.weight.dt[weight_idx] -= ( eta * this.deltaWeight.dt[weight_idx] + net.learningRate * 
                                                                                                    L2lambda * this.weight.dt[weight_idx]);
                            weight_idx++;
                        }
                    }
                }
            }
            Assert(weight_idx == this.weight.dt.length);
            lap.Time();
        }

        /*
            処理時間の計測値のHTML文字列を返す。

            :param int cnt: 処理の回数
        */
        processedTime(cnt){
            return "<tr><td>畳み込み層</td>" + meanProcessedTime([ this.forwardTime[0] ].concat(this.backwardTime).concat(this.updateTime[0]), cnt) + "</tr>";
        }
    }

    /*
        Maxプーリング層のクラス
    */
    class MaxPoolingLayer extends Layer {
        /*        
            :param int filter_size: フィルターのサイズ
        */
        constructor(filter_size) {
            super();
            this.filterSize = filter_size;
        }

        /*
            初期処理
    
            :param Layer prev_layer: 直前のレイヤー 
        */
        init(prev_layer) {
            super.init(prev_layer);

            Assert(this.prevLayer instanceof ConvolutionalLayer, "Pooling-Layer-init");

            this.numChannels = this.prevLayer.numChannels;
            this.numRows = this.prevLayer.numRows / this.filterSize;
            this.numCols = this.prevLayer.numCols / this.filterSize;

            Assert(Math.ceil(this.numRows) == this.numRows && Math.ceil(this.numCols) == this.numCols);

            this.unitSize = this.numChannels * this.numRows * this.numCols;
        }

        /*
            ミニバッチのサイズが変わった時の処理
        */
        miniBatchSizeChanged(){
            super.miniBatchSizeChanged();

            this.y_ = new ArrayView(miniBatchSize, this.unitSize);
            this.maxRow     = new Int8Array(miniBatchSize * this.unitSize);
            this.maxCol     = new Int8Array(miniBatchSize * this.unitSize);
            this.deltaX     = new ArrayView(miniBatchSize, this.prevLayer.unitSize);
        }

        /*
            順伝播
        */
        forward() {
            var lap = new Lap(this.forwardTime);

            var prev_layer = this.prevLayer;

            var prev_y_dt = prev_layer.y_.dt;

            // 出力先
            var output_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // すべての特徴マップに対し
                for (var channel_idx = 0; channel_idx < this.numChannels; channel_idx++) {

                    // 出力の行に対し
                    for (var r1 = 0; r1 < this.numRows; r1++) {
                        var r0 = r1 * this.filterSize;

                        // 出力の列に対し
                        for (var c1 = 0; c1 < this.numCols; c1++) {
                            var c0 = c1 * this.filterSize;

                            if(inGradientCheck){

                                var r2 = this.maxRow[output_idx];
                                var c2 = this.maxCol[output_idx];
                                var prev_y_idx = batch_idx * prev_layer.unitSize + (channel_idx * prev_layer.numRows + (r0 + r2)) * prev_layer.numCols + (c0 + c2);
                                this.y_.dt[output_idx] = prev_y_dt[prev_y_idx];
                            }
                            else{

                                var max_val = -10000;
                                var max_row, max_col;

                                // フィルターの行に対し
                                for (var r2 = 0; r2 < this.filterSize; r2++) {

                                    // フィルターの列に対し
                                    for (var c2 = 0; c2 < this.filterSize; c2++) {

                                        var prev_y_idx = batch_idx * prev_layer.unitSize + (channel_idx * prev_layer.numRows + (r0 + r2)) * prev_layer.numCols + (c0 + c2);
                                        var val = prev_y_dt[prev_y_idx];
                                        if (max_val < val) {

                                            max_val = val;
                                            max_row = r2;
                                            max_col = c2;
                                        }
                                    }
                                }

                                this.y_.dt[output_idx] = max_val;
                                this.maxRow[output_idx] = max_row;
                                this.maxCol[output_idx] = max_col;
                            }

                            output_idx++;
                        }
                    }
                }
            }

            Assert(output_idx == this.y_.dt.length);
            lap.Time();
        }

        /*
            誤差逆伝播
        */
        backpropagation() {
            var lap = new Lap(this.backwardTime);

            var prev_layer = this.prevLayer;

            Assert(this.y_.dt.length == this.nextLayer.deltaX.dt.length);

            for(var i = 0; i < this.deltaX.dt.length; i++){
                this.deltaX.dt[i] = 0;
            }
            lap.Time();

            // 出力先
            var output_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // すべての特徴マップに対し
                for (var channel_idx = 0; channel_idx < this.numChannels; channel_idx++) {
                    var offset = batch_idx * prev_layer.unitSize + channel_idx * prev_layer.numRows * prev_layer.numCols;

                    // 出力の行に対し
                    for (var r1 = 0; r1 < this.numRows; r1++) {
                        var r0 = r1 * this.filterSize;

                        // 出力の列に対し
                        for (var c1 = 0; c1 < this.numCols; c1++) {
                            var c0 = c1 * this.filterSize;

                            var delta = this.nextLayer.deltaX.dt[output_idx];
                            if(delta != 0){

                                var r2 = this.maxRow[output_idx] | 0;
                                var c2 = this.maxCol[output_idx] | 0;
                                var prev_y_idx = offset + (r0 + r2) * prev_layer.numCols + (c0 + c2);
                                this.deltaX.dt[prev_y_idx] += delta;
                            }

                            output_idx++;
                        }
                    }
                }
            }

            Assert(output_idx == this.nextLayer.deltaX.dt.length);
            lap.Time();
        }

        /*
            処理時間の計測値のHTML文字列を返す。

            :param int cnt: 処理の回数
        */
        processedTime(cnt){
            return "<tr><td>Maxプーリング層</td>" + meanProcessedTime([ this.forwardTime[0], null, null, null, this.backwardTime[0], null ], cnt) + "</tr>";
        }
    }


    /*
        ドロップアウト層のクラス
    */
    class DropoutLayer extends Layer {
        /*        
            :param double: ドロップアウトをする確率
        */
        constructor(drop_ratio) {
            super();
            this.dropRatio = drop_ratio;
        }

        /*
            初期処理
    
            :param Layer prev_layer: 直前のレイヤー 
        */
        init(prev_layer) {
            super.init(prev_layer);
            this.unitSize = prev_layer.unitSize;
        }

        /*
            ミニバッチのサイズが変わった時の処理
        */
        miniBatchSizeChanged(){
            super.miniBatchSizeChanged();

            this.y_ = new ArrayView(miniBatchSize, this.unitSize);
            this.deltaX     = new ArrayView(miniBatchSize, this.unitSize);
            this.valid      = new Int8Array(miniBatchSize * this.unitSize);
        }

        /*
            順伝播
        */
        forward() {
            var lap = new Lap(this.forwardTime);

            for(var i = 0; i < this.y_.dt.length; i++){
                if(net.isTraining){
                    // トレーニング データの場合

                    if(this.dropRatio <= Math_random()){
                        // ドロップアウトしない場合

                        this.valid[i]   = 1;
                        this.y_.dt[i] = this.prevLayer.y_.dt[i];
                    }
                    else{
                        // ドロップアウトする場合

                        this.valid[i]   = 0;
                        this.y_.dt[i] = 0;
                    }
                }
                else{
                    // テストデータの場合

                    this.y_.dt[i] = (1 - this.dropRatio) *  this.prevLayer.y_.dt[i];
                }
            }

            lap.Time();
        }

        /*
            誤差逆伝播
        */
        backpropagation() {
            var lap = new Lap(this.backwardTime);

            for(var i = 0; i < this.y_.dt.length; i++){
                if(this.valid[i] == 1){
                    // ドロップアウトしなかった場合

                    this.deltaX.dt[i] = this.nextLayer.deltaX.dt[i];
                }
                else{
                    // ドロップアウトした場合

                    this.deltaX.dt[i] = 0;
                }
            }

            lap.Time();
        }

        /*
            処理時間の計測値のHTML文字列を返す。

            :param int cnt: 処理の回数
        */
        processedTime(cnt){
            return "<tr><td>ドロップアウト層</td>" + meanProcessedTime([ this.forwardTime[0], null, null, null, this.backwardTime[0], null ], cnt) + "</tr>";
        }
    }

    /*
        ニューラルネットワークのクラス
    */
    class NeuralNetwork {
        /*        
            :param GPGPU gpgpu: GPGPUのオブジェクト
        */
        constructor(gpgpu) {
            net = this;
            WebGL2 = gpgpu;
            this.trainingCost = [];
            this.testCost = [];
            this.trainingAccuracy = [];
            this.testAccuracy = [];
        }

        /*        
            :param Layer[] layers: レイヤーの配列
        */
        setLayers(layers){
            this.layers = layers;
            this.lastLayer = layers[layers.length - 1];

            var prev_layer = null;
            for(let layer of layers) {
                layer.init(prev_layer);
                prev_layer = layer;
            }
        }

        /*        
            入力層を作って返す。

            :param int channel_size: チャネル数
            :param int rows:         行数
            :param int cols:         列数
        */
        InputLayer(channel_size, rows, cols){
            return new InputLayer(channel_size, rows, cols);
        }

        /*        
            全結合層を作って返す。

            :param int size: 出力のニューロンの数
            :param int activation_function: 活性化関数のid
        */
        FullyConnectedLayer(size, activation_function){
            return new FullyConnectedLayer(size, activation_function);
        }

        /*        
            畳み込み層を作って返す。

            :param int filter_size: フィルターのサイズ
            :param int channel_size: チャネル数
            :param int activation_function: 活性化関数のid
        */
        ConvolutionalLayer(filter_size, channel_size, activation_function){
            return new ConvolutionalLayer(filter_size, channel_size, activation_function);
        }

        /*        
            Maxプーリング層を作って返す。

            :param int filter_size: フィルターのサイズ
        */
        MaxPoolingLayer(filter_size){
            return new MaxPoolingLayer(filter_size);
        }

        /*        
            ドロップアウト層を作って返す。

            :param double: ドロップアウトをする確率
        */
        DropoutLayer(drop_ratio){
            return new DropoutLayer(drop_ratio);
        }

        /*
            ArrayViewから指定されたインデックスのデータを抜き出して返す。

            :param ArrayView data: 抽出元
            :param int[]     idx_list: インデックスの配列
            :param int       idx_start: 抽出するデータの開始位置
            :param int       idx_cnt: 抽出するデータの数
        */
        ExtractArrayView(data, idx_list, idx_start, idx_cnt) {
            // 1個分のデータのサイズ
            var element_size = data.shape.slice(1).reduce((x, y) => x * y);

            // 抽出元のArrayViewの各次元の要素数
            var shape = data.shape.slice();

            // 戻り値のArrayViewの最初の次元の要素数をidx_cntにする。
            shape[0] = idx_cnt;

            // 戻り値のArrayViewを作る。
            var X = new ArrayView(shape);

            // コピー先の位置
            var dst = 0;

            // idx_startの位置からidx_cnt個のデータをコピーする。
            for (var idx = idx_start; idx < idx_start + idx_cnt; idx++) {
                // 抽出元の位置
                var src = idx_list[idx] * element_size;

                // 1個分のデータをコピーする。
                for (var i = 0; i < element_size; i++) {
                    X.dt[dst] = data.dt[src];
                    src++;
                    dst++;
                }
            }

            return X;
        }

        /*
            正解数を返す。

            :param ArrayView Y: 正解
            :returns: 正解数
        */
        CorrectCount(Y){
            var result = this.lastLayer.y_;

            var ok_cnt = 0;
            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {
                var max_idx = argmax(result.Row(batch_idx));
                if(Y.dt[batch_idx * this.lastLayer.unitSize + max_idx] == 1){
                    ok_cnt++;
                }
            }

            return ok_cnt;
        }

        /*
            最小二乗誤差の微分

            :param float[] last_delta_y_dt: 最後のレイヤーのδy
            :param float[] last_y:          最後のレイヤーのy
            :param float[] batch_Y:         正解の出力
        */
        LeastSquaresDelta(last_delta_y_dt, last_y, batch_Y) {
            for(var i = 0; i < last_delta_y_dt.length; i++){
                last_delta_y_dt[i] = last_y[i] - batch_Y[i];
            }
        }

        /*
            損失関数の微分

            :param float[] last_delta_y_dt: 最後のレイヤーのδy
            :param float[] last_y:          最後のレイヤーのy
            :param float[] batch_Y:         正解の出力
            :param float[] exp_work:        作業用データ
            :param int     range_len:       出力の次元
            :param int     batch_idx:       ミニバッチ内のインデックス
        */
        SoftMax(last_delta_y_dt, last_y, batch_Y, exp_work, range_len, batch_idx) {
            var cost_sum = 0;

            // last_yの最大値を探す。
            var max_val = -10000;
            for (var i = 0; i < range_len; i++) {
                var k = batch_idx * range_len + i;

                if (max_val < last_y[k]) {
                    max_val = last_y[k];
                }
            }

            var sum = 0;
            for (var i = 0; i < range_len; i++) {
                var k = batch_idx * range_len + i;

                var d = Math.exp(last_y[k] - max_val);
                if(! isFinite (d) || d <= 0){
                    d = 0.0000001;
                }
                sum += d;
                exp_work[i] = d;
            }

            for (var i = 0; i < range_len; i++) {
                var k = batch_idx * range_len + i;

                var y = exp_work[i] / sum;
                last_delta_y_dt[k] = y - batch_Y[k];

                var log_y = Math.log(y);
                if(! isFinite (log_y)){
                    continue;
                }

                cost_sum += (batch_Y[k] * log_y);

                if(! isFinite(cost_sum)){
                    Assert(false);
                }
            }

            return - cost_sum;
        }

        /*        
            順伝播と損失関数の計算

            :param float[] batch_Y: 正解の出力
            :param float[] exp_work: 作業用データ
            :param int     batch_idx: ミニバッチ内のインデックス
            :param int     layer_idx: レイヤーのインデックス
            :param float[] last_delta_y_dt: 最後のレイヤーのδy
        */
        forwardCost(batch_Y, exp_work, batch_idx, layer_idx, last_delta_y_dt) {
            var last_layer = this.layers[this.layers.length - 1];

            for(; layer_idx < this.layers.length; layer_idx++){
                this.layers[layer_idx].forward();
            }

            return this.SoftMax(last_delta_y_dt, last_layer.y_.dt, batch_Y, exp_work, last_layer.unitSize, batch_idx);
        }

        /*
            パラメータごとの勾配の計算のチェック

            :param string  name: パラメータ名
            :param float[] params: パラメータの配列
            :param float[] delta_params: パラメータのδの配列 
            :param float[] batch_Y: 正解の出力
            :param float[] exp_work: 作業用データ
            :param double  cost: コスト
            :param int     batch_idx: ミニバッチ内のインデックス
            :param int     layer_idx: レイヤーのインデックス
            :param float[] last_delta_y_dt: 最後のレイヤーのδy
        */
        paramGradientCheck(name, params, delta_params, batch_Y, exp_work, cost, batch_idx, layer_idx, last_delta_y_dt){
            Assert(params.length == delta_params.length);
            // delta bias
            var idx_list = random.RandomSampling(params.length, 3);

            for(let i of idx_list){
                var b = params[i];
                var db = delta_params[i];
                var eps = b * 0.01;

                params[i] = b - eps;
                var cost1 = this.forwardCost(batch_Y, exp_work, batch_idx, layer_idx, last_delta_y_dt);

                params[i] = b + eps;
                var cost2 = this.forwardCost(batch_Y, exp_work, batch_idx, layer_idx, last_delta_y_dt);

                var diff = db * 2 * eps - (cost2 - cost1);
                console.log("delta-%s : %f dC:%f eps:%f cost:%f,%f,%f", name, diff, db, eps, cost1, cost - cost1, cost2 - cost);

                params[i] = b;
            }
        }

        /*        
            勾配の計算のチェック

            :param float[] batch_Y: 正解の出力
            :param float[] exp_work: 作業用データ
            :param float[] costs: ミニバッチ内のコストの配列
        */
        netGradientCheck(batch_Y, exp_work, costs){
            inGradientCheck = true;

            var last_layer = this.layers[this.layers.length - 1];
            var last_delta_y = new Float32Array(last_layer.deltaY.dt);

            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++){
                last_layer.deltaY.dt = new Float32Array(last_delta_y.length);

                for(var i = 0; i < last_layer.unitSize; i++){
                    var k = batch_idx * last_layer.unitSize + i;
                    last_layer.deltaY.dt[k] = last_delta_y[k];
                }

                for (var i = this.layers.length - 1; 1 <= i; i--) {
                    this.layers[i].backpropagation();
                }

                for(var layer_idx = 0; layer_idx < this.layers.length; layer_idx++){

                    console.log("勾配確認 %s", this.layers[layer_idx].constructor.name);
                    this.layers[layer_idx].gradientCheck(batch_Y, exp_work, costs[batch_idx], batch_idx, layer_idx);
                }
            }

            inGradientCheck = false;
        }

        /*
            値が0の比率と負の比率の文字列を返す。

            :param string    name: 変数名 
            :param ArrayView v:    値
        */
        countZero(name, v){
            if(!v){
                return "";
            }

            var cnt = 0;
            var neg = 0;
            for(var i = 0; i < v.dt.length; i++){
                if(v.dt[i] == 0){
                    cnt++;
                }
                if(v.dt[i] < 0){
                    neg++;
                }
            }

            return " " + name + " " + (100.0 * cnt / v.dt.length).toFixed(1) + "/" + (100.0 * neg / v.dt.length).toFixed(1) + " " + v.dt.length;
        }

        /*
            値が0の比率と負の比率をログ出力する。
        */
        logZero(){
            for (let l of this.layers) {
                console.log(
                    l.constructor.name
                    + this.countZero("z"  , l.z_)
                    + this.countZero("y"  , l.y_)
                    + this.countZero("B"  , l.bias)
                    + this.countZero("W"  , l.weight)
                    + this.countZero("δy", l.deltaY)
                    + this.countZero("δz", l.deltaZ)
                    + this.countZero("δB", l.deltaBias)
                    + this.countZero("δW", l.deltaWeight)
                );
            }
        }

        /*        
            SGD(Stochastic Gradient Descent) 確率的勾配降下法

            :param int epochs: エポック数
            :param int mini_batch_size: ミニバッチのサイズ
            :param double learning_rate: 学習率
        */
        * SGD(training_data, test_data, epochs, mini_batch_size, learning_rate) {
            this.learningRate = learning_rate;
            var last_layer = this.layers[this.layers.length - 1];
            last_layer.deltaY = new ArrayView(mini_batch_size, last_layer.unitSize);
            var exp_work = new Float32Array(last_layer.unitSize);
            var change_mini_batch_size = false;

            // 前回yieldでブラウザに制御を戻した時間
            var last_yield_time = undefined;

            for (this.epochIdx = 0; this.epochIdx < epochs; this.epochIdx++) {

                var start_epoch_time = new Date();
                var total_data_cnt = training_data.X.shape[0] + test_data.X.shape[0];
                var total_processed_data_cnt = 0

                // トレーニングとテストの処理
                for(var mode = 0; mode < 2; mode++){
                    this.isTraining = (mode == 0);

                    // 入力(X)と出力(Y)のペア
                    var XY_data;

                    // 正解数
                    var ok_cnt = 0;

                    var cost_sum = 0;

                    if(this.isTraining){

                        XY_data = training_data;
                        miniBatchSize = mini_batch_size;
                    }
                    else{

                        XY_data = test_data;
                        miniBatchSize = (change_mini_batch_size ? 10 : 1) * mini_batch_size;
                    }
                    this.miniBatchSize = miniBatchSize;

                    // ミニバッチ内のコスト
                    var costs = new Float32Array(miniBatchSize);

                    if(change_mini_batch_size || this.epochIdx == 0 && this.isTraining){

                        this.layers.forEach(x => x.miniBatchSizeChanged());
                    }

                    var data_cnt = XY_data.X.shape[0];

                    // 0からdata_cnt-1までの数をシャッフルした配列
                    var idx_list = random.RandomSampling(data_cnt);

                    DataCnt = data_cnt;
                    this.miniBatchCnt = Math.floor(data_cnt / miniBatchSize);

                    for (miniBatchIdx = 0; miniBatchIdx < this.miniBatchCnt; miniBatchIdx++) {

                        var start_pos = miniBatchIdx * miniBatchSize;

                        // idx_listのstart_posからminiBatchSize個のインデックスを使って、XY_dataから入力(X)と出力(Y)のデータを抜き出す。
                        var X = this.ExtractArrayView(XY_data.X, idx_list, start_pos, miniBatchSize);
                        var Y = this.ExtractArrayView(XY_data.Y, idx_list, start_pos, miniBatchSize);

                        // 最初のレイヤー(入力層)の出力にXをセットする。
                        this.layers[0].y_ = X;

                        // 順伝播
                        this.layers.forEach(x => x.forward());

                        if(useSoftMax){

                            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++){

                                costs[batch_idx] = this.SoftMax(last_layer.deltaY.dt, last_layer.y_.dt, Y.dt, exp_work, last_layer.unitSize, batch_idx);
                            }

                            cost_sum += costs.reduce((x,y) => x + y) / miniBatchSize;
                        }
                        else{

                            this.LeastSquaresDelta(last_layer.deltaY.dt, last_layer.y_.dt, Y.dt);
                        }

                        if(this.isTraining){

                            if(useGradientCheck){

                                // 勾配の計算のチェック
                                this.netGradientCheck(Y.dt, exp_work, costs);
                            }
                            else{

                                // 誤差逆伝播
                                for (var i = this.layers.length - 1; 1 <= i; i--) {
                                    this.layers[i].backpropagation();
                                }
                            }

                            // パラメータの更新
                            this.layers.forEach(x => x.updateParameter());
                        }

                        // 処理データ数
                        this.processedDataCnt = (miniBatchIdx + 1) * miniBatchSize;

                        // 正解数
                        ok_cnt += this.CorrectCount(Y);

                        // 正解率
                        var accuracy = ok_cnt   / this.processedDataCnt;

                        // コストの平均
                        var avg_cost = cost_sum / this.processedDataCnt;

                        if(this.isTraining){

                            this.trainingCost[this.epochIdx] = avg_cost;
                            this.trainingAccuracy[this.epochIdx] = accuracy;
                        }
                        else{

                            this.testCost[this.epochIdx] = avg_cost;
                            this.testAccuracy[this.epochIdx] = accuracy;
                        }

                        total_processed_data_cnt += miniBatchSize;

                        if (last_yield_time == undefined || 10 * 1000 < new Date() - last_yield_time) {
                            // 最初か、10秒経過した場合

                            if(last_yield_time != undefined){

                                this.epochTime = Math.round( (new Date() - start_epoch_time) * total_data_cnt / (60 * 1000 * total_processed_data_cnt) );
                            }

                            // ミニバッチごとの処理時間 (レイヤー別)
                            this.processedTimeLayer = this.layers.slice(1).map(layer => layer.processedTime(miniBatchIdx + 1)).join("\n");

                            last_yield_time = new Date();

                            yield 1;
                        }
                    }

                    var sum_last_layer_y = last_layer.y_.dt.reduce((x,y) => x + y);
                    console.log("乱数の数:%d 出力の和:%f", MersenneTwisterIdx, sum_last_layer_y);

                    if(change_mini_batch_size){
                        this.layers.forEach(x => x.clear());
                    }
                }

                console.log("Epoch %d  %.02f% %dmin", this.epochIdx, 100 * this.testAccuracy[this.epochIdx], this.epochTime);

                yield 2;
            }

            if(! change_mini_batch_size){
                this.layers.forEach(x => x.clear());
            }

            yield 0;
        }
    }

    /*        
        シグモイド関数の微分

        :param double z:  
    */
    function sigmoid_prime(z) {
        var f = sigmoid(z);
        return f * (1 - f);
    }

    /*        
        シグモイド関数

        :param double z:  
    */
    function sigmoid(z){
        return 1.0 / (1.0 + Math.exp(-z));
    }

    return new NeuralNetwork(gpgpu);
}
