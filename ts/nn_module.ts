import { gl, GPGPU, UI3D, Package, TextureInfo, range, chk } from "./gpgpu.js";
import { Tensor } from "./tensor.js";

// let gl : WebGL2RenderingContext;

let gpgpu: GPGPU2;

let sz = 1.0;
let vertices = [
    // Front face
    -sz, -sz,  -0.0,
     sz, -sz,  -0.0,
    -sz,  sz,  -0.0,
     sz,  sz,  -0.0,
];

function setRandom(v: Float32Array){
    for(let i = 0; i < v.length; i++){
        v[i] = 2 * Math.random() - 1;
    }
}

// 



function gpuConv2d(x: Tensor, weight: Tensor){
    let [N, iC, H, W] = x.shape
    let [N2, oC, iC2, kH, kW] = weight.shape;

    let buf_h = oC * H;
    let buf_w = W;

    let ConvolutionalLayer = `
    precision highp sampler3D;
    
    uniform sampler3D weight;
    uniform sampler3D x;
    
    out vec4 color;
    
    void main() {
        int iy = int(floor(gl_FragCoord.y));
        int ix = int(floor(gl_FragCoord.x));

    
        int out_channel_idx = iy / ${H};        
        int r1 = iy - out_channel_idx * ${H};

        int c1 = ix;
    
        float sum = 0.0f;
        for(int in_channel_idx = 0; in_channel_idx < ${iC}; in_channel_idx++) {
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
    
                        vec4 txl = texelFetch(x, ivec3(c3, r3, in_channel_idx), 0);
    
                        sum += txl.r * w[c2];
                    }
                }
            }
        }
    
        color = vec4(float(ix), float(iy), sum, 0.0);
    }`;

    console.assert(N == N2 && N == 1 && iC == iC2);

    var shader_src = ConvolutionalLayer;

    let pkg = new Package2();
    pkg.id = `new-conv2d`;
    pkg.buf_h = buf_h;
    pkg.buf_w = buf_w;
    pkg.vertexShader = GPGPU.vertexPositionShader;
    pkg.fragmentShader = shader_src;
    pkg.args = {
        "x"     : new TextureInfo("float", [iC, H, W], x.data),
        "weight": new TextureInfo("vec3", [oC, iC, kH], weight.data),
    };

    return pkg;
}

function drawSceneOnLaptopScreen(pkg: Package2) {
    let [buf_h, buf_w] = [pkg.buf_h, pkg.buf_w];


    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    let dt;
    let format;
    let type;

    if(floatFB){

        dt = new Float32Array(4 * buf_w * buf_h);
        if(Red){

            format = gl.RED;
        }
        else{

            format = gl.RGBA;
        }
        type = gl.FLOAT;
    }
    else{

        dt = new Uint8Array(buf_w * buf_h * 4);
        format = gl.RGBA;
        type = gl.UNSIGNED_BYTE;
    }

    gl.bindTexture(gl.TEXTURE_2D, null);

    let start = Date.now();
    gl.readPixels(0, 0, buf_w, buf_h, format, type, dt); chk();
    log(`time :${Date.now() - start}`)    
    
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);


    return dt;
}


function checkConv2(pkg: Package2, x: Tensor, weight: Tensor, dt: Float32Array){
    let [buf_h, buf_w] = [pkg.buf_h, pkg.buf_w];

    let [N, iC, H, W] = x.shape
    let [N2, oC, iC2, kH, kW] = weight.shape;

    let diff = 0;
    let test_cnt = 0;
    for(let iy = 0; iy < buf_h; iy++){
        for(let ix = 0; ix < buf_w; ix++){
            if(1000 / (buf_h * buf_w) <  Math.random()){
                continue;
            }
            test_cnt++;

            let i = 4 * (iy * buf_w + ix);
            let sum2 = dt[i + 2];

            let out_channel_idx = Math.floor(iy / H);
            let r1 = iy - out_channel_idx * H;
    
            let c1 = ix;
        
            let sum = 0.0;
            for(let in_channel_idx = 0; in_channel_idx < iC; in_channel_idx++) {           
                for (let r2 = 0; r2 < kH; r2++) {
                    let r3 = r1 + r2 - 1;
                
                    for (let c2 = 0; c2 < kW; c2++) {
        
                        let c3 = c1 + c2 - 1;
        
                        if(0 <= c3 && c3 < W && 0 <= r3 && r3 < H){
        
                            let xval = x.at(0, in_channel_idx, r3, c3)  

                            let w = weight.at(0, out_channel_idx, in_channel_idx, r2, c2);
        
                            sum += xval * w;
                        }
                    }
                }
            }

            let d = sum - sum2;
            if(0.0002 < Math.abs(d)){

                log(`NG ${out_channel_idx} ${r1} ${c1} c:${sum} ${sum2}`)
            }
            diff = Math.max(diff, Math.abs(d));
        }
    }

    log(`cnt:${test_cnt}/${buf_h * buf_w}  diff: ${diff.toFixed(7)}`);
}

let floatFB = true;
const Red = false;

function log(s){
    console.log(s);
}

let cubeVertexPositionBuffer;

class Package2 extends Package {
    buf_w = 1000;// 1024;
    buf_h = 1000;//2048;
    vertexAttrLoc: number;
    out: Float32Array;
}

function makeMulPackage() : [Package2, Float32Array, Float32Array] {
    let buf_h = 1000;
    let buf_w = 1000;


    let fragment_shader = `
    out vec4 color;
    
    uniform sampler2D A;
    uniform sampler2D B;
    
    void main(void) {
        int row = int(floor(gl_FragCoord.y));
        int col = int(floor(gl_FragCoord.x));
    
        float sum = 0.0f;
        for(int i = 0; i < ${buf_w}; i++) {
    
            // Aのrow行i列の値を取得します。
            vec4 a = texelFetch(A, ivec2(i, row), 0);
    
            // Bのi行col列の値を取得します。
            vec4 b = texelFetch(B, ivec2(col, i), 0);
    
            // a.rとb.rに取得した値が入っています。
            sum += a.r * b.r;
        }
    
        float w = texelFetch(A, ivec2(col, row), 0).r;
    
        color = vec4(float(col), float(row), sum, w);
    }`;

    let A  = new Float32Array(buf_h * buf_w);
    let B  = new Float32Array(buf_h * buf_w);
    setRandom(A);
    setRandom(B);

    let pkg = new Package2();

    pkg.id            = "test";
    pkg.buf_h = buf_h;
    pkg.buf_w = buf_w;
    pkg.vertexShader = GPGPU.vertexPositionShader;
    pkg.fragmentShader = fragment_shader;
    pkg.args = {
        "A": new TextureInfo("float", [buf_h, buf_w], A),
        "B": new TextureInfo("float", [buf_h, buf_w], B),
    };

    return [pkg, A, B];
}

function checkIdx(pkg: Package2, dt: Float32Array){
    let [buf_h, buf_w] = [pkg.buf_h, pkg.buf_w];

    for(let iy = 0; iy < buf_h; iy++){
        for(let ix = 0; ix < buf_w; ix++){

            let i = 4 * (iy * buf_w + ix);
            let x = dt[i];
            let y = dt[i + 1];
            let z = dt[i + 3];
            if(ix != x || iy != y){

                log(`NG IDX x: ${ix} ${x} ${z} y:${iy} ${y}`)
            }
        }
    }

    log(`check idx end: ${buf_h}x${buf_w}`);
}

function runMul(pkg: Package2, A: Float32Array, B: Float32Array){
    let [buf_h, buf_w] = [pkg.buf_h, pkg.buf_w];

    let dt = new Float32Array(4 * buf_w * buf_h);

    let time = 0;
    let C = new Float32Array(buf_h * buf_w);

    let run_cnt = 8;
    for(let idx = 0; idx < run_cnt; idx++){
    

        if(idx == 0){
            let start = Date.now();
            for(let r = 0; r < buf_h; r++){
                for(let c = 0; c < buf_w; c++){
                    let sum = 0;
        
                    for(let k = 0; k < buf_w; k++){
        
                        sum += A[r*buf_w + k] * B[k*buf_w + c];
                    }
        
                    C[r*buf_w + c] = sum;
                }
            }
            log(`CPU time :${Date.now() - start}`)
        }
        else{

            setRandom(A);
            setRandom(B);
        }    

        let start = Date.now();
        gpgpu.setTextureData(pkg);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        gl.readPixels(0, 0, buf_w, buf_h, gl.RGBA, gl.FLOAT, dt); chk();
        time += Date.now() - start;

        if(idx == 0){

            checkIdx(pkg, dt);
            checkMul(pkg, C, dt);
        }
    }
    let msec = Math.round(time/run_cnt);
    let gf = ((buf_h * buf_w * buf_w) / (msec * 1000 * 1000)).toFixed(1);
    log(`GPU time : ${buf_h}x${buf_w} ${gf}GFLOPS ${ msec }msec`);    

}

function checkMul(pkg: Package2, C2: Float32Array, dt: Float32Array){
    let [buf_h, buf_w] = [pkg.buf_h, pkg.buf_w];

    let diff = 0;
    if(Red){

        for(let iy = 0; iy < buf_h; iy++){
            for(let ix = 0; ix < buf_w; ix++){
                let i = iy * buf_w + ix;
                let x = dt[i];

                if(iy < 3 && ix < 3){

                    log(` ${ix} ${x}`)
                }
                if(ix - x != 0){

                    log(` ${ix - x}`)
                }
            }
        }
    }
    else{
    
        for(let iy = 0; iy < buf_h; iy++){
            for(let ix = 0; ix < buf_w; ix++){
                let i = 4 * (iy * buf_w + ix);
                let c = dt[i + 2];

                let c2 = C2[iy * buf_w + ix];

                diff = Math.max(diff, Math.abs(c - c2));
                if(0.0002 < Math.abs(c - c2)){

                    log(`NG c:${c - c2}`)
                }
            }
        }
    }
    log(`diff ${diff.toFixed(7)}`);
}

class GPGPU2 extends GPGPU {

    constructor(canvas: HTMLCanvasElement, ui3d: UI3D = undefined){
        super(canvas, ui3d);
    }

    initBuffers() {
        cubeVertexPositionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexPositionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    }

    makeFramebuffer(pkg: Package2) {
        let [buf_h, buf_w] = [pkg.buf_h, pkg.buf_w];

        pkg.frameBuffer = gl.createFramebuffer(); chk();
        gl.bindFramebuffer(gl.FRAMEBUFFER, pkg.frameBuffer); chk();
    
        pkg.frameBufferTexture = gl.createTexture(); chk();
    
        // 指定した位置のテクスチャをアクティブにする。
        gl.activeTexture(gpgpu.TEXTUREs[ pkg.textures.length ]); chk();
    
        gl.bindTexture(gl.TEXTURE_2D, pkg.frameBufferTexture); chk();
    
        if(floatFB){
    
            if(Red){
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F   , buf_w, buf_h, 0, gl.RED , gl.FLOAT, null); chk();
            }
            else{
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, buf_w, buf_h, 0, gl.RGBA, gl.FLOAT, null); chk();
            }
        }
        else{
    
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, buf_w, buf_h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);chk();
        }
    
        pkg.renderBuffer = gl.createRenderbuffer(); chk();
        gl.bindRenderbuffer(gl.RENDERBUFFER, pkg.renderBuffer); chk();
    
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pkg.frameBufferTexture, 0); chk();
    
        gl.bindTexture(gl.TEXTURE_2D, null); chk();
        gl.bindRenderbuffer(gl.RENDERBUFFER, null); chk();
        gl.bindFramebuffer(gl.FRAMEBUFFER, null); chk();
    }    

    prepare(pkg: Package2){
        this.initBuffers();

        this.makePackage(pkg);

        this.makeTexture(pkg);
        this.setTextureData(pkg);

        pkg.vertexAttrLoc = gl.getAttribLocation(pkg.program, "aVertexPosition"); chk();
        gl.enableVertexAttribArray(pkg.vertexAttrLoc); chk();

        this.makeFramebuffer(pkg);

        gl.bindFramebuffer(gl.FRAMEBUFFER, pkg.frameBuffer); chk();

        gl.viewport(0, 0, pkg.buf_w, pkg.buf_h); chk();
        gl.clear(gl.COLOR_BUFFER_BIT); chk();
    
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexPositionBuffer); chk();
        gl.vertexAttribPointer(pkg.vertexAttrLoc, 3, gl.FLOAT, false, 0, 0); chk();
    }
}

export function gpuBodyOnLoad(){
    let canvas = document.getElementById("webgl2-canvas") as HTMLCanvasElement;
    gpgpu = new GPGPU2(canvas);
 
    {
        let [H, W] = [1024, 1024];
        let [oC, iC, kH, kW] = [ 16, 16, 3, 3 ];

        // COI:[32 x 32(4)] HW:[1024 x 1024] kHW:[3 x 3]
        let x = new Tensor([1, iC, H, W]);
        let weight = new Tensor([1, oC, iC, kH, kW]);

        setRandom(x.data);
        setRandom(weight.data);
        let pkg = gpuConv2d(x, weight);
        gpgpu.prepare(pkg);
        let dt = drawSceneOnLaptopScreen(pkg);
        gpgpu.clear(pkg);

        checkIdx(pkg, dt);
        checkConv2(pkg, x, weight, dt);
    }

    {
        log("\n--------------------------------------------------\n");
        let [pkg, A, B] = makeMulPackage();
        gpgpu.prepare(pkg);
        runMul(pkg, A, B);
        gpgpu.clear(pkg);
    }

}