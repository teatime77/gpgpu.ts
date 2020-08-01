namespace gpgputs {

let i_latent = 0;

function log(s: string){
    console.log(s)
}

function msg(s: string){
    let span = document.getElementById("msg") as HTMLSpanElement;
    span.innerText = s;
}

function range(n: number) : number[]{
    return [...Array(n).keys()];
}

function zip<T1, T2>(a:T1[], b:T2[]) : [T1, T2][]{
    console.assert(a.length == b.length);

    return range(a.length).map(i => [a[i], b[i]]);
}

function sum(v: number[]): number {
    return v.reduce((a,b)=>a + b, 0);
}

function last<T>(v: T[]){
    return v[v.length - 1];
}

export class Tensor {
    data: Float32Array;
    shape: number[];

    constructor(shape: number[], data: Float32Array | undefined = undefined){
        this.shape = shape;
        if(data == undefined){
            this.data = new Float32Array(this.length());
        }
        else{

            this.data  = data;
            console.assert(this.length() == data.length);
        }
    }

    static fromObj(obj: any) : Tensor{
        let length = obj['shape'].reduce((a: number, b: number)=> a * b, 1);
        let data  = new Float32Array(tensorAll, 4 * obj['pos'], length);

        return new Tensor(obj['shape'], data)
    }

    length(): number {
        return this.shape.reduce((a, b)=> a * b, 1);
    }

    view(...shape: number[]) : Tensor {
        return new Tensor(shape, this.data);
    }

    copy() : Tensor {
        return new Tensor(this.shape.slice(), this.data.slice());
    }

    at(...args: number[]) : number {
        let scale = 1;
        let idx = 0;
        let dim_i = zip(this.shape, args).reverse();
        for(let [dim, i] of dim_i ){
            idx   += scale * i;
            scale *= dim;
        }

        return this.data[idx];
    }

    set(val: number, ...args: number[]) : void {
        let scale = 1;
        let idx = 0;
        let dim_i = zip(this.shape, args).reverse();
        for(let [dim, i] of dim_i ){
            idx   += scale * i;
            scale *= dim;
        }

        this.data[idx] = val;
    }

    row(i: number) : Tensor {
        console.assert(this.shape.length == 2);

        let ncol = this.shape[1];
        let data = this.data.slice(i * ncol, (i + 1) * ncol);

        return new Tensor([1, ncol], data);
    }

    last_val() : number {
        return this.at(...this.shape.map(x=>x-1));
    }

    shape_last() : string {
        return `[${this.shape}] ${this.last_val()}`;
    }

    static align(a: Tensor, b: Tensor) : [Tensor, Tensor] {
        if(a.shape.toString() == b.shape.toString()){
            return [a, b];
        }
        if(a.data.length < b.data.length){
            return [a.expand(...b.shape), b];
        }
        else{
            return [a, b.expand(...a.shape)];
        }
    }

    add(t: Tensor | number) : Tensor {
        if(typeof t == 'number'){
            return new Tensor(this.shape, this.data.map(x => x + t));
        }

        let [a, b] = Tensor.align(this, t as Tensor);

        let dt = new Float32Array(a.data.length);
        let a_dt = a.data;
        let b_dt = b.data;
        for(let i = 0; i < a.data.length; i++){
            dt[i] = a_dt[i] + b_dt[i];
        }

        return new Tensor(this.shape, new Float32Array(dt));
    }

    mul(t: Tensor) : Tensor {
        let [a, b] = Tensor.align(this, t);

        let dt = range(a.data.length).map(i => a.data[i] * b.data[i]);

        return new Tensor(a.shape, new Float32Array(dt));
    }

    scale(n: number) : Tensor {
        let dt = this.data.map(x => n * x);

        return new Tensor(this.shape, new Float32Array(dt));
    }

    square() : Tensor {
        return new Tensor(this.shape, this.data.map(x => x * x));
    }

    sqrt() : Tensor {
        return new Tensor(this.shape, this.data.map(x => Math.sqrt(x)));
    }

    map(f:(n:number)=>number): Tensor {
        return new Tensor(this.shape, this.data.map(x => f(x)));
    }

    expandSub(shape: number[], dst: Tensor, nest: number, this_idx: number, dst_idx: number) : [number, number] {
        console.assert(this.shape[nest] == shape[nest] || this.shape[nest] == 1);

        let this_idx_save = this_idx;
        for(let i = 0; i < shape[nest]; i++){
            if(this.shape[nest] == 1){
                this_idx = this_idx_save;
            }

            if(nest == shape.length - 1){

                dst.data[dst_idx] = this.data[this_idx];
                dst_idx++;
                this_idx++;
            }
            else{

                [this_idx, dst_idx] = this.expandSub(shape, dst, nest + 1, this_idx, dst_idx);
            }
        }
    
        return [this_idx, dst_idx];
    }

    expand(...shape: number[]) : Tensor{
        const startTime = Date.now();
        console.assert(this.shape.length == shape.length);

        let dst = new Tensor(shape);

        let this_idx = 0;
        let dst_idx  = 0;
        [this_idx, dst_idx] = this.expandSub(shape, dst, 0, this_idx, dst_idx);
        console.assert(this_idx == this.data.length && dst_idx == dst.data.length);

        let sec = (Date.now() - startTime) / 1000;
        if(0.5 < sec){

            log(`    expand [${this.shape}]->[${shape}] ${sec}秒`)
        }

        return dst;
    }

    sumSub(dim: number[], dst: Tensor, nest: number, this_idx: number, dst_idx: number) : [number, number] {
        let dst_dims = range(this.shape.length).filter(i => ! dim.includes(i));
        let is_last_dst_dim = (last(dst_dims) == nest);

        let dst_idx_save = dst_idx;
        for(let i = 0; i < this.shape[nest]; i++){
            if(! is_last_dst_dim){
                dst_idx = dst_idx_save;
            }

            if(nest == this.shape.length - 1){

                dst.data[dst_idx] += this.data[this_idx];
                this_idx++;
            }
            else{
    
                [this_idx, dst_idx] = this.sumSub(dim, dst, nest + 1, this_idx, dst_idx);
            }

            if(is_last_dst_dim){
                // log(`INC ${nest} ${i} ${dst_idx}`)
                dst_idx++;
            }
        }
    
        return [this_idx, dst_idx];
    }

    sum(dim:number[]) : Tensor {
        let dst_shape = range(this.shape.length).filter(i => ! dim.includes(i)).map(i => this.shape[i]);
        let dst = new Tensor(dst_shape);
        dst.data.fill(0);

        let this_idx = 0;
        let dst_idx  = 0;
        [this_idx, dst_idx] = this.sumSub(dim, dst, 0, this_idx, dst_idx);
        console.assert(this_idx == this.data.length && dst_idx == dst.data.length);

        return dst;
    }

    printSub(nest: number, ...idxes:number[]){
        let indexes = range(this.shape[nest]);

        if(nest == this.shape.length - 1){
            let s = indexes.map(i => this.at(... idxes.concat([i])).toFixed(7) ).join(', ');
            log(`${" ".repeat(4 * nest)}[ ${s} ]`);
        }
        else{
            log(`${" ".repeat(4 * nest)}[`);
            
            for(let i of indexes){
                this.printSub(nest + 1, ...idxes.concat([i]));
            }

            log(`${" ".repeat(4 * nest)}]`);
        }
    }
    print() : void {
        this.printSub(0);
    }
}

class Module {
    name: string;
    type: string;
    obj : any;
    gpuShape: string = undefined;
    gpuTime: number = undefined;
    nCalc: number = undefined;

    constructor(obj:any){        
        this.name = obj['name'];
        this.type = obj['type'];
        this.obj  = obj;

        console.assert(this.constructor.name == 'Conv2d' || this.constructor.name == 'ConvTranspose2d' || this.constructor.name == obj['type']);

        if(! (this instanceof ModuleListSequential) && !(this instanceof ImageGenerator)){

            allModules[this.name] = this;
        }
    }

    *forward2(x: [Tensor,Tensor]) : Generator<Tensor | undefined> {        
        console.assert(false, '未実装');
    }

    *forward(x: Tensor) : Generator<Tensor | undefined> {  
        console.assert(false, '未実装');
        yield x as Tensor;
    }

    diff(y: Tensor, name: string = 'y') : string {
        let y2 = this.obj[name];

        console.assert(y.shape.toString() == Array.from(y2['shape']).toString());
        let y_last_val = y.last_val();
        let max_y = Math.max(Math.abs(y_last_val), Math.abs(y2['last']));
        let diff  = Math.abs(y_last_val - y2['last']) / max_y;
        // return `[${y.shape}] ${diff.toFixed(7)}`;
        return `${diff.toFixed(7)}`;
    }

    shortType() : string {
        switch(this.type){
        case "PixelwiseNormalization": return "norm";
        case "EqualizedFullyConnect": return "fc";
        case "AddChannelwiseBias": return "bias";
        case "Amplify": return "scale";
        case "LeakyReLU": return "LReLU";
        case "TruncationTrick": return "truncation";
        case "EqualizedModulatedConv2d": return "conv2d";
        case "PixelwiseNoise": return "noise";
        case "EqualizedModulatedConvTranspose2d": return "convT2d";
        case "FusedBlur3x3": return "blur";
        case "ImageGenerator": return "generator";
        }
        return this.type;
    }
}

class LeakyReLU extends Module {
    negative_slope: number;

    constructor(obj:any){
        super(obj);
        this.negative_slope = obj['negative_slope'];
        // log(`${this.name} ${this.type} ${this.negative_slope}`)
    }

    *forward(x: Tensor) : Generator<Tensor | undefined>  {
        running = this as Module;
        yield;

        let dt = new Float32Array(x.data);
        for(let i = 0; i < dt.length; i++){
            if(dt[i] < 0){

                dt[i] *= this.negative_slope;
            }
        }

        let y  = new Tensor(x.shape, dt);

        yield y;
    }
}

class PixelwiseNormalization extends Module {
    constructor(obj:any){
        super(obj);
    }

    * forward(x: Tensor){
        running = this as Module;
        yield;

        console.assert(x.shape.length == 2 && x.shape[0] == 1);
        
        let mean = x.data.map(a => a**2).reduce((x,y) => x + y, 0) / x.shape[1];
        let mean_sqrt = Math.sqrt(mean + 1e-8);

        let v = x.data.map(a => a / mean_sqrt);

        let y = new Tensor(x.shape, new Float32Array(v));

        // log(`FW ${this.name} ${this.type} ${this.diff(y)}`)
        yield y;
    }
}


class TruncationTrick extends Module {
    rate: Tensor;
    avg : Tensor;

    constructor(obj:any){
        super(obj);
        this.rate = Tensor.fromObj(obj['rate'])
        this.avg  = Tensor.fromObj(obj['avg'])
    }


    *forward(x: Tensor) {
        running = this as Module;
        yield;

        let [N, O, D] = this.avg.shape;

        console.assert(x.shape.length == 2 && x.shape[0] == N && N == 1);

        let x1 = range(O).map(a => Array.from(x.data));
        let x2 = x1.flat();

        let avg  = this.avg.data;
        let rate = this.rate.data;
        console.assert(avg.length == rate.length && rate.length == x2.length);

        let dt = range(x2.length).map(i => avg[i] + (x2[i] - avg[i]) * rate[i]);

        let y  = new Tensor([N, O, D], new Float32Array(dt));
        
        // log(`FW ${this.name} ${this.type} y:${this.diff(y)}`);

        yield y;
    }
}

class Amplify extends Module {
    rate: number;

    constructor(obj:any){
        super(obj);
        this.rate = obj['rate'];
    }

    *forward(x: Tensor) {
        running = this as Module;
        yield;

        let dt = new Float32Array(x.data);
        for(let i = 0; i < dt.length; i++){
            dt[i] *= this.rate;
        }

        yield new Tensor(x.shape, dt);
    }
}

class AddChannelwiseBias extends Module {
    bias: Tensor;

    constructor(obj:any){
        super(obj);
        this.bias = Tensor.fromObj(obj['bias'])
        // log(`${this.name} ${this.type} ${this.bias.shape_last()}`)
    }

    *forward(x: Tensor) {
        running = this as Module;
        yield;
        const startTime = Date.now();

        console.assert(x.shape.length == this.bias.shape.length);
        let y;
        if(x.data.length == this.bias.data.length){

            y = x.add(this.bias);
        }
        else if(this.bias.data.length < x.data.length){
            y = this.bias.expand(...x.shape).add(x);
        }
        else{
            y = x.expand(...this.bias.shape).add(this.bias);
        }

        // log(`FW ${this.name} ${this.type} y:${this.diff(y)}`)
        yield y;
    }
}

class EqualizedFullyConnect extends Module {
    weight: Tensor;
    param : any;

    constructor(obj:any){
        super(obj);
        this.weight = Tensor.fromObj(obj['weight'])
        // log(`${this.name} ${this.type} ${this.weight.shape_last()}`)
    }

    *forward(x: Tensor) {
        running = this as Module;
        yield;

        console.assert(x.shape[0] == 1);

        let Linear = `
        in float zero;
        
        // 2次元配列のテクスチャ
        uniform sampler2D A;
        uniform sampler2D B;
        
         // 出力変数C
        out float C;
        
        void main() {
            // テクスチャBの行数と列数を取得します。
            // B_sz.yが行数、B_sz.xが列数です。
            ivec2 B_sz = textureSize(B, 0);
        
            // 出力する行列Cの行(row)と列(col)を計算します。
            // gl_VertexIDは入力変数の何番目の要素かを示すシステム変数です。
            int row = gl_VertexID / B_sz.x;
            int col = gl_VertexID % B_sz.x;
        
            // Cのrow行col列の値は、Aのrow行のベクトルとBのcol列のベクトルの内積です。
        
            // 以下のループでベクトルの内積を計算します。
            float sum = 0.0f;
            for(int i = 0; i < B_sz.y; i++) {
        
                // Aのrow行i列の値を取得します。
                vec4 a = texelFetch(A, ivec2(i, row), 0);
        
                // Bのi行col列の値を取得します。
                vec4 b = texelFetch(B, ivec2(col, i), 0);
        
                // a.rとb.rに取得した値が入っています。
                sum += a.r * b.r;
            }
        
            // 入力変数zeroの値は必要ないですが、使用しない変数はコンパイラが除去してしまいエラーになるので形の上だけ使用します。
            // zeroの値は0なので計算結果には影響しません。
            C = sum + zero;
        }`;

        // 出力変数Cは配列のサイズ(2 * 2)を指定して作ります。
        let y = new Tensor([x.shape[0], this.weight.shape[0]], new Float32Array(x.shape[0] * this.weight.shape[0]));

        let zero = new Float32Array(y.data.length);

        this.gpuTime = 0;

        // 計算のパラメータ
        let pkg = new Package({
            // idはプログラム内でユニークであれば何でも構いません。
            id: "TexMulMat",

            // 頂点シェーダの文字列を指定します。
            vertexShader: Linear,

            // 頂点シェーダ内の入力と出力の変数名に値を割り当てます。
            args: {
                // 出力変数Cと同じサイズで中身の値は0の配列
                "zero": zero,

                "A": gpgpu.makeTextureInfo("float", this.weight.shape, this.weight.data),
                "B": gpgpu.makeTextureInfo("float", [x.shape[1], 1], x.data),
                "C": y.data,
            }
        });

        // パラメータを使い計算します。
        let startTime = Date.now(); 
        gpgpu.compute(pkg);
        this.gpuTime += Date.now() - startTime;

        // WebGLのオブジェクトをクリアします。
        pkg.clear();

        // log(`FW ${this.name} ${this.type} x:${x.shape_last()} y:${this.diff(y)}`)
        this.nCalc = y.data.length * (this.weight.shape[1] * x.shape[1]);

        yield y;
    }
}

class PixelwiseNoise extends Module {
    const_noise: Tensor;
    noise: Tensor | null = null;
    noise_scaler: number;

    constructor(obj:any){
        super(obj);
        this.const_noise = Tensor.fromObj(obj['const_noise']);
        this.noise_scaler = obj['noise_scaler'];
    }

    *forward(x: Tensor) {
        running = this as Module;
        yield;

        let [N,C,H,W] = x.shape;

        if(this.noise == null){
            this.noise = new Tensor(x.shape);

            let idx1 = 0;
            for(let i1 = 0; i1 < C; i1++){
                let idx2 = 0;
                for(let i2 = 0; i2 < H; i2++){
                    for(let i3 = 0; i3 < W; i3++){
                        this.noise.data[idx1] = this.const_noise.data[idx2] * this.noise_scaler;
                        idx1++;
                        idx2++;
                    }
                }
            }
        }

        let y = x.add(this.noise);

        yield y;
    }
}

class FusedBlur3x3 extends Module {
    padding : number;
    groups  : number;
    kernel  : Tensor;

    constructor(obj:any){
        super(obj);

        this.padding = obj['padding'];
        this.groups  = obj['groups'];
        this.kernel  = Tensor.fromObj(obj['kernel']);
    }

    *forward(x: Tensor) {
        running = this as Module;
        yield;

        let sum = 0;
        for(let r = 0; r < 3; r++){
            for(let c = 0; c < 3; c++){
                sum += x.at(0, 511, 6 + r, 6 + c) * this.kernel.at(511, 0, r, c) 
            }
        }

        let y = this.gpuConv2dGroup(x, this.kernel, this.padding, this.groups);

        yield y;
    }

    /*
        GPUによる順伝播
    */
   gpuConv2dGroup(x: Tensor, weight: Tensor, padding: number, groups: number) : Tensor {
        let [N, iC, iH, iW] = x.shape
        let [oC, iC2, kH, kW] = weight.shape;

        console.assert(padding == 1 && groups == iC && iC == oC && iC2 == 1);

        let oH = iH + 2 * padding - (kH - 1) - 1 + 1;
        let oW = iW + 2 * padding - (kW - 1) - 1 + 1;

        let shader_src = `
        precision highp sampler3D;
        
        uniform sampler3D weight;
        uniform sampler3D x;
        
        in  float zero;
        out float y;
        
        void main() {
            int idx = int(gl_VertexID);
        
            int channel_idx = idx / (${oH} * ${oW});
            idx -= channel_idx * (${oH} * ${oW});
        
            int r1 = idx / ${oW};
            int c1 = idx - r1 * ${oW};
        
            float sum = 0.0f;
        
            int r2, c2;
        
            for (r2 = 0; r2 < ${kH}; r2++) {
        
                for (c2 = 0; c2 < ${kW}; c2++) {
        
                    // int c3 = c1 + c2 - (${kW} - 1);
                    // int r3 = r1 + r2 - (${kH} - 1);
                    int c3 = c1 + c2 - 1;
                    int r3 = r1 + r2 - 1;
        
                    if(0 <= c3 && c3 < ${iW} && 0 <= r3 && r3 < ${iH}){
        
                        vec4 txl = texelFetch(x    , ivec3(c3, r3, channel_idx), 0);
        
                        vec4  w = texelFetch(weight, ivec3(${kW} - 1 - c2, ${kH} - 1 - r2, channel_idx), 0);
                        // vec4  w = texelFetch(weight, ivec3(c2, r2, channel_idx), 0);
        
                        sum += txl.r * w.r;
                    }
                }
            }
        
            y = sum + zero;
        }`;

        var y = new Tensor([N, oC, oH, oW])
        let zero = new Float32Array(y.data.length);
    

        this.gpuTime = 0;
        let pkg = new Package({
            id : `${this.name}`,
            vertexShader: shader_src,
            args : {
                "zero"  : zero,
                "x"     : gpgpu.makeTextureInfo("float", [iC, iH, iW], x.data),
                "weight": gpgpu.makeTextureInfo("float", [iC, kH, kW], weight.data),
                "y"     : y.data
            }
        });

        let startTime = Date.now(); 
        gpgpu.compute(pkg);
        this.gpuTime += Date.now() - startTime;
        this.nCalc = (oC * oH * oW) * (iC * kH * kW);

        pkg.clear();

        return y;
    }

}

class ConvTranspose2d extends Module {
    fc: EqualizedFullyConnect;
    bias: AddChannelwiseBias;
    weight: Tensor;
    demodulate: boolean;
    padding: number;
    stride: number;
    groups: number;

    constructor(obj:any){
        super(obj);

        this.fc = new EqualizedFullyConnect(obj['fc']);
        this.bias = new AddChannelwiseBias(obj['bias']);
        this.weight = Tensor.fromObj(obj['weight']);
        this.demodulate = obj['demodulate'];
        this.padding = obj['padding'];
        this.stride = obj['stride'];
        this.groups = obj['groups'];
    }

    *forward2(pack: [Tensor, Tensor]){
        let [x, style] = pack;
        let [N, iC, H, W] = x.shape;
        let [N2, oC, iC2, kH, kW] = this.weight.shape;

        console.assert(N == N2 && iC == iC2);

        let fc_y : Tensor | undefined;
        for(fc_y of this.fc.forward(style)) yield;

        let bias_y : Tensor | undefined;
        for(bias_y of this.bias.forward(fc_y!)) yield;

        running = this as Module;
        yield;

        let mod_rates = bias_y!.add(1);// (N, iC)

        let modulated_weight = this.weight.mul(mod_rates.view(N,1,iC,1,1));

        let weight: Tensor;
        if(this.demodulate){
            let demod_norm = modulated_weight.square().sum([2,3,4]).add(1e-8).sqrt().map(x => 1 / x) // (N, oC)
            weight = modulated_weight.mul(demod_norm.view(N, oC, 1, 1, 1)); // (N,oC,iC,kH,kW)

        }
        else{

            weight = modulated_weight;
        }

        x = x.view(1, N * iC, H, W);

        let y : Tensor | undefined;
        for(y of this.gpuConvTranspose2d(x, weight.view(N*oC, iC, kH, kW), this.stride, this.padding)) yield;

        let [dim1, dim2, Hp1, Wp1] = y!.shape;
        y = y!.view(N, oC, Hp1, Wp1);

        yield y;
    }


    /*
        GPUによる順伝播
    */
   *gpuConvTranspose2d(x: Tensor, weight: Tensor, stride: number, padding: number){
        let [N, iC, iH, iW] = x.shape
        let [oC, iC2, kH, kW] = weight.shape;

        console.assert(stride == 2 && padding == 0);

        let oH = (iH - 1) * stride + (kH - 1) + 1;
        let oW = (iW - 1) * stride + (kW - 1) + 1;

        this.nCalc = (oC * oH * oW) * (iC * kH * kW);

        let iCmini = iC;

        let cache_mem = 5 * 1000 * 1000;
        let mem;

        while(true){
            mem = 4 * (iCmini * oH * oW + oC * iCmini * kH * kW );
            if(mem < cache_mem || iCmini == 4 || iCmini % 2 != 0){
                break;
            }
            iCmini /= 2;
        }

        this.gpuShape = `COI:[${oC} x ${iC}(${iCmini})] HW:[${oH} x ${oW}] kHW:[${kH} x ${kW}] mem:${(mem/(1000*1000)).toFixed(1)}`;
        

        let shader_src = `
        precision highp sampler3D;
        
        uniform sampler3D weight;
        uniform sampler3D x;
        uniform int in_channel_base;
        
        in  float zero;
        out float y;
        
        void main() {
            int idx = int(gl_VertexID);
        
            int num_in_rows_ex = 2 * ${iH} - 1;
            int num_in_cols_ex = 2 * ${iW} - 1;
        
            int out_channel_idx = idx / (${oH} * ${oW});
            idx -= out_channel_idx * (${oH} * ${oW});
        
            int r1 = idx / ${oW};
            int c1 = idx - r1 * ${oW};
        
            float sum = 0.0f;
            for(int in_channel_offset = 0; in_channel_offset < ${iCmini}; in_channel_offset++) {
                int in_channel_idx = in_channel_base + in_channel_offset;
            
                for (int r2 = 0; r2 < ${kH}; r2++) {
        
                    for (int c2 = 0; c2 < ${kW}; c2++) {
        
                        int c3 = c1 + c2 - (${kW} - 1);
                        int r3 = r1 + r2 - (${kH} - 1);
        
                        if(0 <= c3 && c3 < num_in_cols_ex && 0 <= r3 && r3 < num_in_rows_ex && c3 % 2 == 0 && r3 % 2 == 0){
                            c3 /= 2;
                            r3 /= 2;
        
                            vec4 txl = texelFetch(x    , ivec3(c3, r3, in_channel_idx), 0);
        
                            vec4  w = texelFetch(weight, ivec3((${kH} - 1 - r2) * ${kW} + (${kW} - 1 - c2), in_channel_idx, out_channel_idx), 0);
                            // vec4  w = texelFetch(weight, ivec3(r2 * ${kW} + c2, out_channel_idx, in_channel_idx), 0);
                            // vec4  w = texelFetch(weight, ivec3( (${kW} - 1 - c2) * ${kH} + (${kH} - 1 - r2), out_channel_idx, in_channel_idx), 0);
        
                            sum += txl.r * w.r;
                        }
                    }
                }
            }
        
            y = sum + zero;
        }`;

        let y  = new Tensor([N, oC, oH, oW]);
        let y2 = new Float32Array(y.data.length);

        let zero    = new Float32Array(y.data.length);

        let pkg = new Package({
            id : `${this.name}`,
            vertexShader: shader_src,
            args : {
                "in_channel_base": 0,
                "zero"  : zero,
                "x"     : gpgpu.makeTextureInfo("float", [iC, iH, iW], x.data),
                "weight": gpgpu.makeTextureInfo("float", [oC, iC, kH * kW], weight.data),
                "y"     : y2
            }
        });

        gpgpu.makePackage(pkg);

        this.gpuTime = 0;

        for(let idx = 0; idx * iCmini < iC; idx++){
            pkg.args["in_channel_base"] = idx * iCmini;

            let startTime = Date.now(); 
            gpgpu.compute(pkg);
            this.gpuTime += Date.now() - startTime;

            for(let i = 0; i < y2.length; i++){
                y.data[i] += y2[i];
            }

            yield;
        }
        pkg.clear();

        yield y;
    }

}

class Conv2d extends Module {
    fc: EqualizedFullyConnect;
    bias: AddChannelwiseBias;
    weight: Tensor;
    demodulate: boolean;
    padding: number;
    stride: number;
    groups: number;

    constructor(obj:any){
        super(obj);

        this.fc = new EqualizedFullyConnect(obj['fc']);
        this.bias = new AddChannelwiseBias(obj['bias']);
        this.weight = Tensor.fromObj(obj['weight']);
        this.demodulate = obj['demodulate'];
        this.padding = obj['padding'];
        this.stride = obj['stride'];
        this.groups = obj['groups'];
    }

    *forward2(pack: [Tensor, Tensor]) {
        let startTime = Date.now();

        let [x, style] = pack;
        let [N, iC, H, W] = x.shape
        let [N2, oC, iC2, kH, kW] = this.weight.shape;

        console.assert(N == N2 && iC == iC2);

        let fc_y: Tensor | undefined;
        for(fc_y of this.fc.forward(style)) yield;

        let bias_y : Tensor | undefined;
        for(bias_y of this.bias.forward(fc_y!)) yield;

        running = this as Module;
        yield;

        let mod_rates = bias_y!.add(1);// (N, iC)

        if(1000 < startTime - Date.now()){log(`  A ${this.type} ${(Date.now() - startTime) / 1000}秒`);} startTime = Date.now();

        let modulated_weight = this.weight.mul(mod_rates.view(N,1,iC,1,1));
        let demod_norm;
        let weight: Tensor;

        if(1000 < startTime - Date.now()){log(`  B ${this.type} ${(Date.now() - startTime) / 1000}秒`);} startTime = Date.now();

        if(this.demodulate){
            demod_norm = modulated_weight.square().sum([2,3,4]).add(1e-8).sqrt().map(x => 1 / x) // (N, oC)
            weight = modulated_weight.mul(demod_norm.view(N, oC, 1, 1, 1)); // (N,oC,iC,kH,kW)
            console.assert(true);
        }
        else{

            weight = modulated_weight;
        }

        if(1000 < startTime - Date.now()){log(`  C ${this.type} ${(Date.now() - startTime) / 1000}秒`);} startTime = Date.now();

        let y;
        for(y of this.gpuConv2d(x, weight)) yield;

        yield y;
    }


    /*
        GPUによる順伝播
    */
    *gpuConv2d(x: Tensor, weight: Tensor) {
        let [N, iC, H, W] = x.shape
        let [N2, oC, iC2, kH, kW] = weight.shape;

        console.assert(N == N2 && N == 1 && iC == iC2);

        this.nCalc = (oC * H * W) * (iC * kH * kW);

        let iCmini = iC;

        let cache_mem = 5 * 1000 * 1000;
        let mem;

        while(true){
            mem = 4 * (iCmini * H * W + oC * iCmini * kH * kW );
            if(mem < cache_mem || iCmini == 4 || iCmini % 2 != 0){
                break;
            }
            iCmini /= 2;
        }

        this.gpuShape = `COI:[${oC} x ${iC}(${iCmini})] HW:[${H} x ${W}] kHW:[${kH} x ${kW}] mem:${(mem/(1000*1000)).toFixed(1)}`;

        console.assert(kH == 3 && kW == 3 || kH == 1 && kW == 1);

        let weight2_shape;
        var shader_src;
        let weight2_texelType;

        let shiftRowsCols = Math.log2(H * W);
        let shiftCols = Math.log2(W);
        console.assert(Math.round(shiftRowsCols) == shiftRowsCols && Math.round(shiftCols) == shiftCols);

        if(kH == 3){

            weight2_texelType   = "vec3";
            weight2_shape = [oC, iC, kH];

            shader_src = `
            precision highp sampler3D;
            
            uniform sampler3D weight;
            uniform sampler3D x;
            uniform int in_channel_base;
            
            in  float zero;
            out float y;
            
            void main() {
                int idx = int(gl_VertexID);
            
                // int out_channel_idx = idx / (${H} * ${W});
                int out_channel_idx = idx >> ${shiftRowsCols};
                idx -= out_channel_idx * (${H} * ${W});
            
                // int r1 = idx / ${W};
                int r1 = idx >> ${shiftCols};
                int c1 = idx - r1 * ${W};
            
                float sum = 0.0f;
                for(int in_channel_offset = 0; in_channel_offset < ${iCmini}; in_channel_offset++) {
                    int in_channel_idx = in_channel_base + in_channel_offset;
            
                    vec4 ww[3];
                    for (int r2 = 0; r2 < ${kH}; r2++){
                        ww[r2]  = texelFetch(weight, ivec3(r2, in_channel_idx, out_channel_idx), 0);
                    }
                
                    for (int r2 = 0; r2 < ${kH}; r2++) {
                        int r3 = r1 + r2 - 1;
            
                        vec4 w = ww[r2];
            
                        for (int c2 = 0; c2 < ${kW}; c2++) {
            
                            int c3 = c1 + c2 - 1;
            
                            if(0 <= c3 && c3 < ${W} && 0 <= r3 && r3 < ${H}){
            
                                vec4 txl = texelFetch(x    , ivec3(c3, r3, in_channel_idx), 0);
                        
                                sum += txl.r * w[c2];
                            }
                        }
                    }
                }
            
                y = sum + zero;
            }`;
        }
        else{

            weight2_texelType   = "float";
            weight2_shape = [oC, iC];

            shader_src = `
            precision highp sampler3D;
            
            uniform sampler2D weight;
            uniform sampler3D x;
            uniform int in_channel_base;
            
            in  float zero;
            out float y;
            
            void main() {
                int idx = int(gl_VertexID);
            
                int out_channel_idx = idx / (${H} * ${W});
                idx -= out_channel_idx * (${H} * ${W});
            
                int r1 = idx / ${W};
                int c1 = idx - r1 * ${W};
            
                float sum = 0.0f;
                for(int in_channel_offset = 0; in_channel_offset < ${iCmini}; in_channel_offset++) {
                    int in_channel_idx = in_channel_base + in_channel_offset;
                
                    vec4 txl = texelFetch(x    , ivec3(c1, r1, in_channel_idx), 0);
            
                    vec4  w = texelFetch(weight, ivec2(in_channel_idx, out_channel_idx), 0);
            
                    sum += txl.r * w.r;
                }
            
                y = sum + zero;
            }`;
        }

        let y  = new Tensor([N, oC, H, W]);
        let y2 = new Float32Array(y.data.length);

        let zero    = new Float32Array(y.data.length);

        let pkg = new Package({
            id : `${this.name}`,
            vertexShader: shader_src,
            args : {
                "in_channel_base": 0,
                "zero"  : zero,
                "x"     : gpgpu.makeTextureInfo("float", [iC, H, W], x.data),
                "weight": gpgpu.makeTextureInfo(weight2_texelType, weight2_shape, weight.data),
                "y"     : y2
            }
        });

        gpgpu.makePackage(pkg);

        this.gpuTime = 0;
        for(let idx = 0; idx * iCmini < iC; idx++){
            pkg.args["in_channel_base"] = idx * iCmini;

            let startTime = Date.now(); 
            gpgpu.compute(pkg);
            this.gpuTime += Date.now() - startTime;
        
            for(let i = 0; i < y2.length; i++){
                y.data[i] += y2[i];
            }

            yield;
        }

        pkg.clear();

        yield y;
    }
}

class ImageGenerator extends Module {
    const_input: Tensor;
    mapping: Sequential;
    blocks: ModuleList;
    toRGBs: ModuleList;

    constructor(obj:any){
        super(obj);
        this.const_input = Tensor.fromObj(obj['const_input'])

        // log(`${this.const_input.at(0, 1,2,3)}`);

        this.mapping = new Sequential(obj['mapping']);
        this.blocks  = new ModuleList(obj['blocks']);
        this.toRGBs  = new ModuleList(obj['toRGBs']);        
    }

    *genImage(latent: Tensor) {
        let styles : Tensor | undefined;
        for(styles of this.mapping.forward(latent)) yield;

        styles = styles!.view(...styles!.shape.slice(1));

        let tmp : Tensor | undefined;
        for(tmp of this.blocks.modules[0].forward2( [this.const_input, styles.row(0) ] )) yield;

        let skip : Tensor | undefined;
        for(skip of this.toRGBs.modules[0].forward2( [tmp!, styles.row(1)] )) yield;

        // log(`FW ${this.name} ${this.type} ${styles.shape} skip:${skip.shape_last()}`)

        for(let idx of range(this.toRGBs.modules.length - 1)){
            let convU = this.blocks.modules[1 + 2 * idx];
            let convF = this.blocks.modules[2 + 2 * idx];
            let toRGB = this.toRGBs.modules[1 + idx];
            let styU  = styles.row(1 + 2 * idx);
            let styF  = styles.row(2 + 2 * idx);
            let styT  = styles.row(3 + 2 * idx);

            for(tmp of convU.forward2( [tmp!,styU] )) yield;

            for(tmp of convF.forward2( [tmp!,styF] )) yield;

            let rgb : Tensor | undefined;
            for(rgb of toRGB.forward2( [tmp!,styT] )) yield;

            let up_skip = interpolate(skip!);
            skip = rgb!.add(up_skip);
            // log(`FW ${this.name} ${this.type} tmp:${tmp.shape_last()}`);

            yield skip;
        }
    }
}

class ModuleListSequential extends Module {
    modules: Module[];

    constructor(obj:any){
        super(obj);

        this.modules = obj['modules'].map((x: any) => parseModel(x));
    }

    *forward(x: Tensor) {
        let y: Tensor | undefined = x;
        for(let m of this.modules){
            let startTime = Date.now(); // 開始時間

            for(y of m.forward(y!)) yield;

            if(m.gpuTime != undefined){
                gpuTimeAll += m.gpuTime;
                nCalcAll   += m.nCalc;
            }

            let sec = (Date.now() - startTime) / 1000;
            if(m instanceof ConvTranspose2d && 37000000 < m.nCalc){

                let py_time = (m.obj['time'] == undefined ? "" :  m.obj['time'].toFixed(3));
                let gpu_shape = (m.gpuShape == undefined ? "" : m.gpuShape);
                let gpu_time = "";
                if(m.gpuTime != undefined){
                    gpu_time = `${py_time}|${(m.gpuTime / 1000).toFixed(3)}秒 ${(m.nCalc / (10000*10000)).toFixed(3)}億回 ${((m.nCalc / m.gpuTime) / (1000 * 1000)).toFixed(3)}GFLOPS`;
                }

                let diff = (use_random || i_latent != 0 || m.obj['y'] == undefined ? "" : `diff:${m.diff(y!)}`);
                log(`FW ${m.shortType()} ${gpu_shape} ${gpu_time} ${diff}`);
            }
        }

        yield y;
    }
}

class ModuleList extends ModuleListSequential {
}

class Sequential extends ModuleListSequential {
}

let modelObj: any;
let generator: ImageGenerator;
let tensorAll : ArrayBuffer;
let gpgpu: GPGPU;
let allModules : { [name: string]: Module }  = {};
let running : Module | null = null;
let use_random = false;
let nCalcAll = 0;
let gpuTimeAll = 0;

function parseModel(obj:any) : Module{
    switch(obj['type']){
    case 'PixelwiseNormalization':
        return new PixelwiseNormalization(obj);

    case 'TruncationTrick':
        return new TruncationTrick(obj);

    case 'Amplify':
        return new Amplify(obj);

    case 'AddChannelwiseBias':
        return new AddChannelwiseBias(obj);

    case 'EqualizedFullyConnect':
        return new EqualizedFullyConnect(obj);
            
    case 'PixelwiseNoise':
        return new PixelwiseNoise(obj);

    case 'FusedBlur3x3':
        return new FusedBlur3x3(obj);

    case 'EqualizedModulatedConvTranspose2d':
        return new ConvTranspose2d(obj);

    case 'EqualizedModulatedConv2d':
        return new Conv2d(obj);

    case 'ModuleList':
        return new ModuleList(obj);

    case 'Sequential':
        return new Sequential(obj);

    case 'LeakyReLU':
        return new LeakyReLU(obj);

    default:
        throw new Error();
    }
}

function luminosity(f: number){
    return (Math.max(-1, Math.min(f, 1)) + 1) / 2 * 255;
}


function putImage(t: Tensor){
    let canvas = document.getElementById('canvas-2d') as HTMLCanvasElement;
    let ctx = canvas.getContext('2d')!;

    let [N, C, H, W] = t.shape;

    let imageData = ctx.createImageData(W, H);
    let data = imageData.data;

    let idx = 0;
    let r_base = 0;
    let g_base = H * W;
    let b_base = 2 * H * W;
    for(let y = 0; y < H; y++){
        for(let x = 0; x < W; x ++){
            let dst = idx * 4;

            data[dst    ] = luminosity(t.data[r_base + idx]); // red
            data[dst + 1] = luminosity(t.data[g_base + idx]); // green
            data[dst + 2] = luminosity(t.data[b_base + idx]); // blue                
            data[dst + 3] = 255;

            idx++;
        }    
    }

    ctx.putImageData(imageData, 0, 0);
}

function main(generator: ImageGenerator, latents: Tensor){
    makeModuleTable();

    let prev_running : Module | null = null;
    let running_yield = 0;
    nCalcAll = 0;
    gpuTimeAll = 0;
    
    let latent = latents.row(0);
    if(use_random){

        latent.data = latent.data.map(a => 2 * Math.random() - 1);
    }
    let gen = generator.genImage(latent);
    let img_idx = 1;
    let timer = setInterval(()=>{
        let data = gen.next();

        if(data.done){
            prev_running = null;
            running = null;                

            let s = `${i_latent} ${Math.round(nCalcAll / (10000*10000))}億 ${(gpuTimeAll/1000).toFixed(1)}秒 ${((nCalcAll/gpuTimeAll) / (1000 * 1000)).toFixed(3)}GFLOPS`;
            msg(s);
            log(s);

            let check = document.getElementById("auto-download") as HTMLInputElement;
            if(check.checked){

                // https://stackoverflow.com/questions/10673122/how-to-save-canvas-as-an-image-with-canvas-todataurl
                let canvas = document.getElementById('canvas-2d') as HTMLCanvasElement;
                var link = document.getElementById('download-link')!;
                let dt = new Date();
                if(use_random){

                    link.setAttribute('download', `gan-${dt.getFullYear()}-${dt.getMonth() + 1}-${dt.getDate()}-${dt.getHours()}-${dt.getMinutes()}.png`);
                }
                else{
                    link.setAttribute('download', `gan-standard-${i_latent}.png`);

                }
                link.setAttribute('href', canvas.toDataURL("image/png").replace("image/png", "image/octet-stream"));
                link.click();
                img_idx++;
            }

            clearModuleTable();

            i_latent++;
            if(use_random){
                latent.data = latent.data.map(a => 2 * Math.random() - 1);

            }
            else{

                if(i_latent == latents.shape[0]){

                    clearInterval(timer);
                    return;
                }

                latent = latents.row(i_latent);
            }

            nCalcAll = 0;
            gpuTimeAll = 0;
            gen = generator.genImage(latent);

        }
        else{
            if(running != prev_running){

                if(prev_running != null){

                    drawModule(prev_running, "blue");
                }
                running_yield = 0;
                drawModule(running!, "white", "red");

                prev_running = running;
            }
            else{
                running_yield++;
                if(running_yield % 2 == 0){

                    drawModule(running!, "white", "red");
                }
                else{

                    drawModule(running!, "red", "white");
                }
            }

            if(data.value != undefined){
                drawModule(prev_running!, "blue");

                let skip = data.value as Tensor;
                putImage(skip);
            }
        }
    }, 1)
}

export function tensorBodyOnLoad(){
    gpgpu = CreateGPGPU();

    msg("ダウンロード 開始");
    fetchJson('data/model.json')
    .then(obj => {
        msg("JSON ダウンロード 終了");
        modelObj = obj;
    });

    fetchArray('data/model.bin')
    .then((v) => {
        msg(`モデル ダウンロード 終了 length:${v.byteLength}`);

        tensorAll = v;

        let latents   = Tensor.fromObj(modelObj['latents']);

        generator = new ImageGenerator(modelObj['generator']);

        main(generator, latents);
    });

}

function fetchJson(path: string) {
    // Promiseクラスのインスタンスを関数の戻り値にする
    // Promiseクラスのコンストラクタの引数には関数を渡す
    // その関数は、resolve関数とreject関数を引数に取り、戻り値は無し
    return new Promise(function(resolve:(text:string)=>void) {

        let k = window.location.href.lastIndexOf("/");

        let url = `${window.location.href.substring(0, k)}/${path}`;

        const url2 = encodeURI(url);
        log(`fetch-text:${url}`);
        fetch(url2)
        .then((res: Response) => {
            return res.json();
        })
        .then( (data: any) => {
            resolve(data);
        })
        .catch((error: any) => {
            console.error('Error:', error);
        });
    });
}


function fetchArray(path: string) {
    // Promiseクラスのインスタンスを関数の戻り値にする
    // Promiseクラスのコンストラクタの引数には関数を渡す
    // その関数は、resolve関数とreject関数を引数に取り、戻り値は無し
    return new Promise(function(resolve:(text:ArrayBuffer)=>void) {

        let k = window.location.href.lastIndexOf("/");

        let url = `${window.location.href.substring(0, k)}/${path}`;

        const url2 = encodeURI(url);
        log(`fetch-array:${url}`);
        fetch(url2)
        .then((res: Response) => {
            return res.arrayBuffer();
        })
        .then( (data: ArrayBuffer) => {
            // resolve(new Float32Array(data));
            resolve(data);
        })
        .catch((error: any) => {
            console.error('Error:', error);
        });
    });
}

function interpolate(src: Tensor): Tensor {
    console.assert(src.shape[0] == 1 && src.shape[1] == 3);

    let h_src = src.shape[2];
    let w_src = src.shape[3];

    let y_shape = [src.shape[0], src.shape[1], 2 * src.shape[2], 2 * src.shape[3] ];
    let dst = new Tensor(y_shape);
    let h = dst.shape[2];
    let w = dst.shape[3];
    let idx = 0;
    let ry = (h_src - 1) / (h - 1);
    let rx = (w_src - 1) / (w - 1);
    for(let c = 0; c < 3; c++){
        for(let i =0; i < h; i++){
            let iy1 = Math.floor(i * ry);
            let iy2 = Math.min(iy1 + 1, h_src - 1);
            let py  = (i * ry) - iy1;

            for(let j = 0; j < w; j++){
                let ix1 = Math.floor(j * rx);
                let ix2 = Math.min(ix1 + 1, w_src - 1);
                let px  = (j * rx) - ix1;

                let f11 = src.at(0, c, iy1, ix1);
                let f12 = src.at(0, c, iy1, ix2);
                let f21 = src.at(0, c, iy2, ix1);
                let f22 = src.at(0, c, iy2, ix2);

                let f1 = f11 * (1 - px) + f12 * px;
                let f2 = f21 * (1 - px) + f22 * px;

                let f = f1 * (1 - py) + f2 * py;

                dst.data[idx] = f;
                idx++;
            }
        }
    }

    return dst;
}

function drawModule(mod: Module, color: string, bg_color: string = "white"){
    let cell = document.getElementById(mod.name) as HTMLTableCellElement;
    cell.style.color = color;
    cell.style.backgroundColor = bg_color;
}

function clearModuleTable(){
    let tbl = document.getElementById("module-tbl") as HTMLTableElement;
    for(let cell of tbl.getElementsByTagName("td")){
        (cell as HTMLTableCellElement).style.color = "black";
    }
}

function makeModuleTable(){
    let names = [
        [ "mapping.0", "mapping.1", "mapping.2", "mapping.3", "mapping.4" ], 
        [ "mapping.5", "mapping.6", "mapping.7", "mapping.8" ], 
        [ "mapping.9", "mapping.10", "mapping.11", "mapping.12" ],
        [ "mapping.13", "mapping.14", "mapping.15", "mapping.16" ], 
        [ "mapping.17", "mapping.18", "mapping.19", "mapping.20" ], 
        [ "mapping.21", "mapping.22", "mapping.23", "mapping.24" ], 
        [ "mapping.25", "mapping.26", "mapping.27", "mapping.28" ], 
        [ "mapping.29", "mapping.30", "mapping.31", "mapping.32" ], 
        [ "mapping.33" ], 
        [ "blocks.0.0.fc", "blocks.0.0.bias", "blocks.0.0", "blocks.0.1", "blocks.0.2", "blocks.0.3", "blocks.0.4", "", "toRGBs.0.0.fc", "toRGBs.0.0.bias", "toRGBs.0.0", "toRGBs.0.1" ], 

        [ "blocks.1.0.fc", "blocks.1.0.bias", "blocks.1.0", "blocks.1.1", "blocks.1.2", "blocks.1.3", "blocks.1.4", "blocks.1.5" ], 
        [ "blocks.2.0.fc", "blocks.2.0.bias", "blocks.2.0", "blocks.2.1", "blocks.2.2", "blocks.2.3", "blocks.2.4", "", "toRGBs.1.0.fc", "toRGBs.1.0.bias", "toRGBs.1.0", "toRGBs.1.1" ], 

        [ "blocks.3.0.fc", "blocks.3.0.bias", "blocks.3.0", "blocks.3.1", "blocks.3.2", "blocks.3.3", "blocks.3.4", "blocks.3.5" ], 
        [ "blocks.4.0.fc", "blocks.4.0.bias", "blocks.4.0", "blocks.4.1", "blocks.4.2", "blocks.4.3", "blocks.4.4", "", "toRGBs.2.0.fc", "toRGBs.2.0.bias", "toRGBs.2.0", "toRGBs.2.1" ], 

        [ "blocks.5.0.fc", "blocks.5.0.bias", "blocks.5.0", "blocks.5.1", "blocks.5.2", "blocks.5.3", "blocks.5.4", "blocks.5.5" ], 
        [ "blocks.6.0.fc", "blocks.6.0.bias", "blocks.6.0", "blocks.6.1", "blocks.6.2", "blocks.6.3", "blocks.6.4", "", "toRGBs.3.0.fc", "toRGBs.3.0.bias", "toRGBs.3.0", "toRGBs.3.1" ], 

        [ "blocks.7.0.fc", "blocks.7.0.bias", "blocks.7.0", "blocks.7.1", "blocks.7.2", "blocks.7.3", "blocks.7.4", "blocks.7.5" ], 
        [ "blocks.8.0.fc", "blocks.8.0.bias", "blocks.8.0", "blocks.8.1", "blocks.8.2", "blocks.8.3", "blocks.8.4", "", "toRGBs.4.0.fc", "toRGBs.4.0.bias", "toRGBs.4.0", "toRGBs.4.1" ], 

        [ "blocks.9.0.fc", "blocks.9.0.bias", "blocks.9.0", "blocks.9.1", "blocks.9.2", "blocks.9.3", "blocks.9.4", "blocks.9.5" ], 
        [ "blocks.10.0.fc", "blocks.10.0.bias", "blocks.10.0", "blocks.10.1", "blocks.10.2", "blocks.10.3", "blocks.10.4", "", "toRGBs.5.0.fc", "toRGBs.5.0.bias", "toRGBs.5.0", "toRGBs.5.1" ], 

        [ "blocks.11.0.fc", "blocks.11.0.bias", "blocks.11.0", "blocks.11.1", "blocks.11.2", "blocks.11.3", "blocks.11.4", "blocks.11.5" ], 
        [ "blocks.12.0.fc", "blocks.12.0.bias", "blocks.12.0", "blocks.12.1", "blocks.12.2", "blocks.12.3", "blocks.12.4", "", "toRGBs.6.0.fc", "toRGBs.6.0.bias", "toRGBs.6.0", "toRGBs.6.1" ], 

        [ "blocks.13.0.fc", "blocks.13.0.bias", "blocks.13.0", "blocks.13.1", "blocks.13.2", "blocks.13.3", "blocks.13.4", "blocks.13.5" ], 
        [ "blocks.14.0.fc", "blocks.14.0.bias", "blocks.14.0", "blocks.14.1", "blocks.14.2", "blocks.14.3", "blocks.14.4", "", "toRGBs.7.0.fc", "toRGBs.7.0.bias", "toRGBs.7.0", "toRGBs.7.1" ], 

        [ "blocks.15.0.fc", "blocks.15.0.bias", "blocks.15.0", "blocks.15.1", "blocks.15.2", "blocks.15.3", "blocks.15.4", "blocks.15.5" ], 
        [ "blocks.16.0.fc", "blocks.16.0.bias", "blocks.16.0", "blocks.16.1", "blocks.16.2", "blocks.16.3", "blocks.16.4", "", "toRGBs.8.0.fc", "toRGBs.8.0.bias", "toRGBs.8.0", "toRGBs.8.1" ]
    ];

    let tbl = document.getElementById("module-tbl") as HTMLTableElement;
    let img_size = 4;
    for(let row of names){
        // let tr = document.createElement("tr");
        let tr = tbl.insertRow(-1);
        for(let [i, name] of row.entries()){
            if(name == ""){
                tr.insertCell();
                continue;
            }

            let mod = allModules[name];

            if(i == 0 && name.startsWith("blocks.")){
                let n = parseInt(name.split('.')[1]);
                if(n == 0 || n % 2 == 1){

                    let cell1 = tr.insertCell();
                    cell1.innerText = `${img_size} x ${img_size}`;
                    cell1.style.borderWidth = "3px";
                    cell1.style.borderStyle = "ridge";

                    if(n != 0){
                        cell1.rowSpan = 2;
                    }
                    
                    img_size *= 2;
                }                
            }

            let cell = tr.insertCell();
            cell.id = name;
            cell.innerText = mod.shortType();            
            cell.style.borderWidth = "3px";
            cell.style.borderStyle = "ridge";
        }
    }
}

}
