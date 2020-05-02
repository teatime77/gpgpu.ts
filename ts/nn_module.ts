import { gl, GPGPU, UI3D, Package, TextureInfo, range, chk } from "./gpgpu.js";
import { Tensor } from "./tensor.js";

// let gl : WebGL2RenderingContext;

let gpgpu: GPGPU2;

let sz = 1.0;
let vertices = [
    // Front face
    -sz, -sz,  -0.0,
     sz, -sz,  -0.0,
     sz,  sz,  -0.0,
    -sz,  sz,  -0.0,
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
    
    in vec4 vPosition;
    out vec4 color;
    
    void main() {
        int ix = int(round(${buf_w}.0 * (1.0 + vPosition.x - ${1.0 / buf_w}) / 2.0));
        int iy = int(round(${buf_h}.0 * (1.0 + vPosition.y - ${1.0 / buf_h}) / 2.0));
    
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
    pkg.vertexShader = vertex_shader;
    pkg.fragmentShader = shader_src;
    pkg.args = {
        "x"     : new TextureInfo("float", [iC, H, W], x.data),
        "weight": new TextureInfo("vec3", [oC, iC, kH], weight.data),
        "aVertexPosition": new Float32Array(vertices)
    };

    return pkg;
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
            if(0.0001 < Math.abs(d)){

                log(`NG ${out_channel_idx} ${r1} ${c1} c:${sum} ${sum2}`)
            }
            diff = Math.max(diff, Math.abs(d));
        }
    }

    log(`cnt:${test_cnt}/${buf_h * buf_w}  diff: ${diff}`);
}


let vertex_shader = `
in vec3 aVertexPosition;

out vec2 vTextureCoord;
out vec3 vTransformedNormal;
out vec4 vPosition;

void main(void) {
    vPosition   = vec4(aVertexPosition, 1.0);
    gl_Position = vec4(aVertexPosition, 1.0);
}`;

function initGL() {
    let canvas = document.getElementById("webgl2-canvas") as HTMLCanvasElement;

    try {
        // gl = canvas.getContext("webgl2") as WebGL2RenderingContext;
        const ext = gl.getExtension("EXT_color_buffer_float");
        if (!ext) {
            alert("need EXT_color_buffer_float");
            return;
        }

    } catch (e) {
    }
    if (!gl) {
        alert("Could not initialise WebGL, sorry :-(");
    }
}

let renderbuffer;
let rttFramebuffer;
let floatFB = true;
const Red = false;

function log(s){
    console.log(s);
}

let cubeVertexPositionBuffer;
let cubeVertexIndexBuffer;



class Package2 extends Package {
    buf_w = 1000;// 1024;
    buf_h = 1000;//2048;
    vertexAttrLoc: number;
    out: Float32Array;
}

function makeMulPackage() : [Package2, Float32Array, Float32Array] {
    let buf_h = 2000;
    let buf_w = 2000;


    let fragment_shader = `
    in vec4 vPosition;
    out vec4 color;
    
    uniform sampler2D A;
    uniform sampler2D B;
    
    void main(void) {
        float x = ${buf_w}.0 * (1.0 + vPosition.x - ${1.0 / buf_w}) / 2.0;
        float y = ${buf_h}.0 * (1.0 + vPosition.y - ${1.0 / buf_h}) / 2.0;
    
        int row = int(round(y));
        int col = int(round(x));
    
        float sum = 0.0f;
        for(int i = 0; i < ${buf_w}; i++) {
    
            // Aのrow行i列の値を取得します。
            vec4 a = texelFetch(A, ivec2(i, row), 0);
    
            // Bのi行col列の値を取得します。
            vec4 b = texelFetch(B, ivec2(col, i), 0);
    
            // a.rとb.rに取得した値が入っています。
            sum += a.r * b.r;
        }
    
        // float w = texelFetch(A, ivec2(0, 2), 0).r * texelFetch(B, ivec2(2, 0), 0).r
        //         + texelFetch(A, ivec2(1, 2), 0).r * texelFetch(B, ivec2(2, 1), 0).r
        //         + texelFetch(A, ivec2(2, 2), 0).r * texelFetch(B, ivec2(2, 2), 0).r;
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
    pkg.vertexShader = vertex_shader;
    pkg.fragmentShader = fragment_shader;
    pkg.args = {
        "A": new TextureInfo("float", [buf_h, buf_w], A),
        "B": new TextureInfo("float", [buf_h, buf_w], B),
        "aVertexPosition": new Float32Array(vertices)
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
            if(ix != x || iy != y){

                log(`NG IDX x:${ix - x} y:${iy - y}`)
            }
        }
    }

    log(`check idx end: ${buf_h}x${buf_w}`);
}

function runMul(pkg: Package2, A: Float32Array, B: Float32Array){
    let [buf_h, buf_w] = [pkg.buf_h, pkg.buf_w];

    let dt = new Float32Array(4 * buf_w * buf_h);

    gl.bindFramebuffer(gl.FRAMEBUFFER, rttFramebuffer); chk();

    gl.viewport(0, 0, buf_w, buf_h); chk();
    gl.clear(gl.COLOR_BUFFER_BIT); chk();

    gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexPositionBuffer); chk();
    gl.vertexAttribPointer(pkg.vertexAttrLoc, cubeVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0); chk();

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeVertexIndexBuffer); chk();


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
        gl.drawElements(gl.TRIANGLES, cubeVertexIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0); chk();

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
                if(0.0001 < Math.abs(c - c2)){

                    log(`NG c:${c - c2}`)
                }
            }
        }
    }
    log(`diff ${diff}`);
}

class GPGPU2 extends GPGPU {
    constructor(canvas: HTMLCanvasElement, ui3d: UI3D = undefined){
        super(canvas, ui3d);
    }

    initBuffers() {
        cubeVertexPositionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexPositionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        cubeVertexPositionBuffer.itemSize = 3;
        cubeVertexPositionBuffer.numItems = 4;
    
        cubeVertexIndexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeVertexIndexBuffer);
        let cubeVertexIndices = [0, 1, 2, 0, 2, 3];
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(cubeVertexIndices), gl.STATIC_DRAW);
        cubeVertexIndexBuffer.itemSize = 1;
        cubeVertexIndexBuffer.numItems = 6;
    }

    makeFramebuffer(pkg: Package2) {
        let [buf_h, buf_w] = [pkg.buf_h, pkg.buf_w];

        rttFramebuffer = gl.createFramebuffer(); chk();
        gl.bindFramebuffer(gl.FRAMEBUFFER, rttFramebuffer); chk();
    
        let tex = gl.createTexture(); chk();
    
        // 指定した位置のテクスチャをアクティブにする。
        gl.activeTexture(gpgpu.TEXTUREs[ pkg.textures.length ]); chk();
    
        gl.bindTexture(gl.TEXTURE_2D, tex); chk();
    
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
    
        renderbuffer = gl.createRenderbuffer(); chk();
        gl.bindRenderbuffer(gl.RENDERBUFFER, renderbuffer); chk();
    
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0); chk();
    
        gl.bindTexture(gl.TEXTURE_2D, null); chk();
        gl.bindRenderbuffer(gl.RENDERBUFFER, null); chk();
        gl.bindFramebuffer(gl.FRAMEBUFFER, null); chk();
    }    

    drawSceneOnLaptopScreen(pkg: Package2) {
        let [buf_h, buf_w] = [pkg.buf_h, pkg.buf_w];

        gl.bindFramebuffer(gl.FRAMEBUFFER, rttFramebuffer); chk();
    
        gl.viewport(0, 0, buf_w, buf_h); chk();
        gl.clear(gl.COLOR_BUFFER_BIT); chk();
    
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexPositionBuffer); chk();
        gl.vertexAttribPointer(pkg.vertexAttrLoc, cubeVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0); chk();
    
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeVertexIndexBuffer); chk();
    
        gl.drawElements(gl.TRIANGLES, cubeVertexIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0); chk();
    
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

    prepare(pkg: Package2){
        this.initBuffers();

        this.makePackage(pkg);

        this.makeTexture(pkg);
        this.setTextureData(pkg);

        pkg.vertexAttrLoc = gl.getAttribLocation(pkg.program, "aVertexPosition"); chk();
        gl.enableVertexAttribArray(pkg.vertexAttrLoc); chk();

        this.makeFramebuffer(pkg);
    }
}

export function gpuBodyOnLoad(){
    let canvas = document.getElementById("webgl2-canvas") as HTMLCanvasElement;
    gpgpu = new GPGPU2(canvas);

    initGL();

    if(false){
        let [H, W] = [1024, 1024];
        let [oC, iC, kH, kW] = [ 16, 16, 3, 3 ];

        // COI:[32 x 32(4)] HW:[1024 x 1024] kHW:[3 x 3]
        let x = new Tensor([1, iC, H, W]);
        let weight = new Tensor([1, oC, iC, kH, kW]);

        setRandom(x.data);
        setRandom(weight.data);
        let pkg = gpuConv2d(x, weight);
        gpgpu.prepare(pkg);
        let dt = this.drawSceneOnLaptopScreen(pkg);
        gpgpu.clear(pkg);

        checkIdx(pkg, dt);
        checkConv2(pkg, x, weight, dt);
    }
    else{

        let [pkg, A, B] = makeMulPackage();
        gpgpu.prepare(pkg);
        runMul(pkg, A, B);
        gpgpu.clear(pkg);
    }
}