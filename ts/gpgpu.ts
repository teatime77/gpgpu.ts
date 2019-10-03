﻿namespace gpgpu {

declare let mat4:any;
declare let mat3:any;

type TextureValue = Float32Array | HTMLImageElement | HTMLCanvasElement;

let gl : WebGL2RenderingContext;

/*
    エラーのチェックをします。
*/
function assert(condition:boolean, message:string|undefined=undefined) {
    if (!condition) {
        throw new Error(message != undefined ? message :"Assertion failed");
    }
}

export function range(n: number) : number[]{
    return [...Array(n).keys()];
}

/*
    WebGLのエラーのチェックをします。
*/
function chk() {
    assert(gl.getError() == gl.NO_ERROR);
}

export class Drawable {
    static count = 0;
    param: PackageParameter;
    transform: Float32Array;

    constructor(){
        this.transform = mat4.create();
        mat4.identity(this.transform);
    }

    getVertexColors(color: Color, numVertex: number) : number[] {
        let vertexColors = [];

        range(numVertex + 1).forEach(x => {
            vertexColors.push(color.r, color.g, color.b, color.a);
        });

        return vertexColors;
    }

    move(x: number, y: number, z: number) : Drawable {
        mat4.translate(this.transform, [x, y, z]);

        return this;
    }

    scale(x: number, y: number, z: number) : Drawable {
        mat4.scale(this.transform, [x, y, z]);

        return this;
    }
    
    getParam() {
        return this.param;
    }
}

export class ComponentDrawable extends Drawable {
    children: Drawable[];

}

export class PackageParameter{
    id: string;
    vertexShader: string;
    fragmentShader: string;
    args: Map<string, Float32Array|TextureInfo>;
    VertexIndexBuffer: Uint16Array | Uint32Array;
    elementCount: number;
}

class Package{
    id: string;
    program: WebGLProgram;
    transformFeedback: WebGLTransformFeedback;
    vertexIndexBufferInf: VertexIndexBufferInf;
    textures: TextureInfo[];
    attribElementCount: number;
    elementCount: number;
    varyings: ArgInf[];
    attributes: ArgInf[];
    uniforms: ArgInf[];
}

/*
    テクスチャ情報

    テクスチャのテクセルの型、サイズ、値の情報を持ちます。
*/
export class TextureInfo {
    name: string;
    isArray: boolean;
    texelType: string;
    samplerType: string;
    Texture: WebGLTexture;
    shape: number[];
    value: TextureValue;
    locTexture: WebGLUniformLocation;

    /*
        TextureInfoのコンストラクタ
    */
    constructor(texel_type: string, shape: number[], value: TextureValue) {
        // テクセルの型
        this.texelType = texel_type;

        // テクスチャのサイズ
        this.shape = shape;

        // テクスチャの値
        this.value = value;
    }
}

class ArgInf {
    name: string;
    value: Float32Array;
    type: string;
    isArray: boolean;
    feedbackBuffer: WebGLBuffer;
    locUniform: WebGLUniformLocation;
    AttribBuffer: WebGLBuffer;
    AttribLoc: number;
}

class VertexIndexBufferInf {
    value: Uint16Array | Uint32Array;
    buffer: WebGLBuffer;
}

export class Mesh {
    vertexPosition: Float32Array;
    vertexNormal: Float32Array;
    vertexColor : Float32Array;
    textureCoord: Float32Array;
    textureImage: TextureInfo;
}

class DrawParam{
    xRot : number;
    yRot : number;
    x    : number;
    y    : number;
    z    : number;
}

/*
    GPGPUのメインのクラス
*/
export class GPGPU {
    static readonly textureSphereVertexShader = `

        const vec3 uAmbientColor = vec3(0.2, 0.2, 0.2);
        const vec3 uLightingDirection =  normalize( vec3(0.25, 0.25, 1) );
        const vec3 uDirectionalColor = vec3(0.8, 0.8, 0.8);

        // 位置
        in vec3 vertexPosition;

        // 法線
        in vec3 vertexNormal;

        // テクスチャ座標
        in vec2 textureCoord;

        uniform mat4 uPMVMatrix;
        uniform mat3 uNMatrix;

        out vec3 vLightWeighting;

        out vec2 uv0;
        out vec2 uv1;

        void main(void) {
            gl_Position = uPMVMatrix * vec4(vertexPosition, 1.0);

            vec3 transformedNormal = uNMatrix * vertexNormal;
            float directionalLightWeighting = max(dot(transformedNormal, uLightingDirection), 0.0);
            vLightWeighting = uAmbientColor +uDirectionalColor * directionalLightWeighting;

            uv0 = fract( textureCoord.st );
            uv1 = fract( textureCoord.st + vec2(0.5,0.5) ) - vec2(0.5,0.5);
        }
    `;

    // GPGPU用のフラグメントシェーダ。(何も処理はしない。)
    static readonly minFragmentShader =
        `out vec4 color;

        void main(){
            color = vec4(1.0);
        }`;

    // デフォルトの動作のフラグメントシェーダ
    static readonly defaultFragmentShader =
        `in vec3 vLightWeighting;
        in vec2 uv0;
        in vec2 uv1;

        uniform sampler2D textureImage;

        out vec4 color;

        void main(void) {
            vec2 uvT;

            uvT.x = ( fwidth( uv0.x ) < fwidth( uv1.x )-0.001 ) ? uv0.x : uv1.x ;
            uvT.y = ( fwidth( uv0.y ) < fwidth( uv1.y )-0.001 ) ? uv0.y : uv1.y ;

            vec4 textureColor = texture(textureImage, uvT);

            color = vec4(textureColor.rgb * vLightWeighting, textureColor.a);
        }
        `;

        static readonly planeTextureVertexShader = `

        const vec3 uAmbientColor = vec3(0.2, 0.2, 0.2);
        const vec3 uLightingDirection =  normalize( vec3(0.25, 0.25, 1) );
        const vec3 uDirectionalColor = vec3(0.8, 0.8, 0.8);

        // 位置
        in vec3 vertexPosition;

        // 法線
        in vec3 vertexNormal;

        // テクスチャ座標
        in vec2 textureCoord;

        uniform mat4 uPMVMatrix;
        uniform mat3 uNMatrix;

        out vec3 vLightWeighting;

        out vec2 uv;

        void main(void) {
            gl_Position = uPMVMatrix * vec4(vertexPosition, 1.0);

            vec3 transformedNormal = uNMatrix * vertexNormal;
            float directionalLightWeighting = max(dot(transformedNormal, uLightingDirection), 0.0);
            vLightWeighting = uAmbientColor +uDirectionalColor * directionalLightWeighting;

            uv = textureCoord;
        }
    `;

    static readonly planeTextureFragmentShader =
        `in vec3 vLightWeighting;
        in vec2 uv;

        uniform sampler2D textureImage;

        out vec4 color;

        void main(void) {
            vec4 textureColor = texture(textureImage, uv);

            color = vec4(textureColor.rgb * vLightWeighting, textureColor.a);
        }
        `;

        static readonly planeVertexShader = `

        const vec3 uAmbientColor = vec3(0.2, 0.2, 0.2);
        const vec3 uLightingDirection =  normalize( vec3(0.25, 0.25, 1) );
        const vec3 uDirectionalColor = vec3(0.8, 0.8, 0.8);

        // 位置
        in vec3 vertexPosition;

        // 法線
        in vec3 vertexNormal;

        // 色
        in vec4 vertexColor;

        uniform mat4 uPMVMatrix;
        uniform mat3 uNMatrix;

        out vec3 vLightWeighting;

        // 色
        out vec4 fragmentColor;

        void main(void) {
            gl_Position = uPMVMatrix * vec4(vertexPosition, 1.0);

            vec3 transformedNormal = uNMatrix * vertexNormal;
            float directionalLightWeighting = max(dot(transformedNormal, uLightingDirection), 0.0);
            vLightWeighting = uAmbientColor +uDirectionalColor * directionalLightWeighting;
            fragmentColor = vertexColor;
        }
    `;

    static readonly planeFragmentShader =
        `in vec3 vLightWeighting;

        in vec4 fragmentColor;

        out vec4 color;

        void main(void) {
            color = vec4(fragmentColor.rgb * vLightWeighting, fragmentColor.a);
        }
        `;


    canvas: HTMLCanvasElement;
    TEXTUREs: number[];
    packages : Map<string, Package>;
    drawables: Drawable[];
    drawParam: DrawParam;

    /*
        GPGPUのコンストラクタ
    */
    constructor(canvas: HTMLCanvasElement) {
        console.log("init WebGL");

        if (!canvas) {
            // canvasが指定されていない場合

            // canvasを作る。
            canvas = document.createElement('canvas');
            
            // canvasをサイズをセットする。
            canvas.width = 32;
            canvas.height = 32;

            // canvasをdocumentに追加する。
            document.body.appendChild(canvas);
        }

        this.canvas = canvas;

        // canvasからWebGL2のcontextを得る。
        gl = canvas.getContext('webgl2', { antialias: false }) as WebGL2RenderingContext;
        var isWebGL2 = !!gl;
        if (!isWebGL2) {
            // WebGL2のcontextを得られない場合

            console.log("WebGL 2 is not available. See How to get a WebGL 2 implementation");
            console.log("https://www.khronos.org/webgl/wiki/Getting_a_WebGL_Implementation");

            throw "WebGL 2 is not available.";
        }

        // パッケージのリストを初期化する。
        this.packages = new Map<string, Package>();

        // 標準のシェーダの文字列をセットする。
        this.setStandardShaderString();

        this.TEXTUREs = [gl.TEXTURE0, gl.TEXTURE1, gl.TEXTURE2, gl.TEXTURE3];
    }

    /*
        WebGLのオブジェクトを返します。
    */
    getGL() {
        return gl;
    }

    /*
        テクスチャ情報を作ります。
    */
    makeTextureInfo(texel_type: string, shape: number[], value: TextureValue) {
        return new TextureInfo(texel_type, shape, value);
    }

    /*
        WebGLのオブジェクトをすべて削除します。
    */
    clearAll() {
        var packages = Object.assign({}, this.packages);
        for (var id in packages) {
            this.clear(id);
        }
    }

    /*
        指定したidのWebGLのオブジェクトをすべて削除します。
    */
    clear(id: string) {
        var pkg = this.packages[id];

        if (pkg) {
            // 指定したidのパッケージがある場合

            delete this.packages[id]

            gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();

            if (pkg.idxBuffer) {

                // バッファを削除する。
                gl.deleteBuffer(pkg.idxBuffer); chk();
            }

            // すべてのvarying変数に対し
            for (let varying of pkg.varyings) {
                if (varying.feedbackBuffer) {
                    // Transform Feedbackバッファがある場合

                    // バッファを削除する。
                    gl.deleteBuffer(varying.feedbackBuffer); chk();
                }
            }

            if (pkg.transformFeedback) {
                // Transform Feedbackがある場合

                gl.deleteTransformFeedback(pkg.transformFeedback); chk();
            }

            // テクスチャのバインドを解く。
            gl.bindTexture(gl.TEXTURE_2D, null); chk();
            gl.bindTexture(gl.TEXTURE_3D, null); chk();

            // すべてのテクスチャを削除する。
            pkg.textures.forEach(x => gl.deleteTexture(x.Texture), chk())

            // プログラムを削除する。
            gl.deleteProgram(pkg.program); chk();
        }
    }

    /*
        シェーダのソースコードを解析します。
    */
    parseShader(pkg: Package, param: PackageParameter) {
        // attribute変数、uniform変数、テクスチャ、varying変数の配列を初期化する。
        pkg.attributes = [];
        pkg.uniforms = [];
        pkg.textures = [];
        pkg.varyings = [];

        // 頂点シェーダとフラグメントシェーダのソースに対し
        for(let shader_text of[ param.vertexShader,  param.fragmentShader ]) {

            // 行ごとに分割する。
            var lines = shader_text.split(/(\r\n|\r|\n)+/);

            // すべての行に対し
            for(let line of lines) {

                // 行を空白で分割する。
                var tokens = line.trim().split(/[\s\t]+/);

                if (tokens.length < 3) {
                    // トークンの長さが3未満の場合
                    continue;
                }

                // 最初、2番目、3番目のトークン
                var tkn0 = tokens[0];
                var tkn1 = tokens[1];
                var tkn2 = tokens[2];

                if (tkn0 != "in" && tkn0 != "uniform" && tkn0 != "out") {
                    // 最初のトークンが in, uniform, out でない場合
                    continue;
                }

                if (shader_text == param.fragmentShader && tkn0 != "uniform") {
                    // フラグメントシェーダで uniform でない場合 ( フラグメントシェーダの入力(in)と出力(out)はアプリ側では使わない。 )

                    continue;
                }
                assert(tkn1 == "int" || tkn1 == "float" || tkn1 == "vec2" || tkn1 == "vec3" || tkn1 == "vec4" ||
                    tkn1 == "sampler2D" || tkn1 == "sampler3D" ||
                    tkn1 == "mat4" || tkn1 == "mat3" || tkn1 == "bool");


                var arg_name;
                var is_array = false;
                var k1 = tkn2.indexOf("[");
                if (k1 != -1) {
                    // 3番目のトークンが [ を含む場合

                    // 配列と見なす。
                    is_array = true;

                    // 変数名を得る。
                    arg_name = tkn2.substring(0, k1);
                }
                else{
                    // 3番目のトークンが [ を含まない場合

                    var k2 = tkn2.indexOf(";");
                    if (k2 != -1) {
                        // 3番目のトークンが ; を含む場合

                        // 変数名を得る。
                        arg_name = tkn2.substring(0, k2);
                    }
                    else{
                        // 3番目のトークンが ; を含まない場合

                        // 変数名を得る。
                        arg_name = tkn2;
                    }
                }

                // 変数の値を得る。
                var arg_val = param.args[arg_name];

                if (arg_val == undefined) {
                    if(tokens[0] == "out"){
                        continue;
                    }
                }

                if (tkn1 == "sampler2D" || tkn1 == "sampler3D") {
                    // テクスチャのsamplerの場合

                    assert(tokens[0] == "uniform" && arg_val instanceof TextureInfo);
                    let tex_inf = arg_val as TextureInfo;

                    // 変数名をセットする。
                    tex_inf.name = arg_name;

                    // samplerのタイプをセットする。
                    tex_inf.samplerType = tkn1;

                    // 配列かどうかをセットする。
                    tex_inf.isArray = is_array;

                    // テクスチャの配列に追加する。
                    pkg.textures.push(tex_inf);
                }
                else {
                    // テクスチャのsamplerでない場合

                    // 変数の名前、値、型、配列かどうかをセットする。
                    var arg_inf = { name: arg_name, value: arg_val, type: tkn1, isArray: is_array } as ArgInf;

                    switch (tokens[0]) {
                        case "in":
                            // attribute変数の場合

                            pkg.attributes.push(arg_inf);
                            break;

                        case "uniform":
                            // uniform変数の場合

                            pkg.uniforms.push(arg_inf);
                            break;

                        case "out":
                            // varying変数の場合

                            pkg.varyings.push(arg_inf);
                            break;
                    }
                }
            }
        }
    }

    /*
        WebGLのプログラムを作ります。
    */
    makeProgram(vertex_shader: WebGLShader, fragment_shader: WebGLShader, varyings: ArgInf[]) : WebGLProgram {
        // プログラムを作る。
        var prg = gl.createProgram(); chk();

        // 頂点シェーダをアタッチする。
        gl.attachShader(prg, vertex_shader); chk();

        // フラグメントシェーダをアタッチする。
        gl.attachShader(prg, fragment_shader); chk();

        if (varyings) {
            // varying変数がある場合

            // varying変数の名前の配列
            var varying_names = varyings.map(x => x.name);

            // Transform Feedbackで使うvarying変数を指定する。
            gl.transformFeedbackVaryings(prg, varying_names, gl.SEPARATE_ATTRIBS); chk();   // gl.INTERLEAVED_ATTRIBS 
        }

        // プログラムをリンクする。
        gl.linkProgram(prg); chk();

        if (!gl.getProgramParameter(prg, gl.LINK_STATUS)) {
            // リンクエラーがある場合

            console.log("Link Error:" + gl.getProgramInfoLog(prg));
        }

        // 頂点シェーダを削除する。
        gl.deleteShader(vertex_shader); chk();

        // フラグメントシェーダを削除する。
        gl.deleteShader(fragment_shader); chk();

        return prg;
    }

    /*        
        シェーダを作ります。
    */
    makeShader(type: number, source: string) : WebGLShader {
        source = "#version 300 es\nprecision highp float;\nprecision highp int;\n" + source;

        // シェーダを作る。
        var shader = gl.createShader(type); chk();

        // シェーダにソースをセットする。
        gl.shaderSource(shader, source); chk();

        // シェーダをコンパイルする。
        gl.compileShader(shader); chk();

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            // コンパイル エラーの場合

            alert(gl.getShaderInfoLog(shader));
            return null;
        }

        return shader;
    }

    /*
        attribute変数を作ります。
    */
    makeAttrib(pkg: Package) {
        // すべてのattribute変数に対し
        for (let attrib of pkg.attributes) {
            // attribute変数の次元
            var attrib_dim = this.vecDim(attrib.type);

            // 要素の個数
            var elemen_count = attrib.value.length / attrib_dim;

            if (pkg.elementCount == undefined) {
                pkg.attribElementCount = elemen_count;
            }
            else {

                assert(pkg.elementCount == elemen_count);
            }

            // バッファを作る。
            attrib.AttribBuffer = gl.createBuffer();

            // attribute変数の位置
            attrib.AttribLoc = gl.getAttribLocation(pkg.program, attrib.name); chk();

            // 指定した位置のattribute配列を有効にする。
            gl.enableVertexAttribArray(attrib.AttribLoc); chk();

            // attribute変数の位置と変数名をバインドする。
            gl.bindAttribLocation(pkg.program, attrib.AttribLoc, attrib.name);
        }
    }

    /*
        テクスチャを作ります。
    */
    makeTexture(pkg: Package) {
        // すべてのテクスチャに対し
        for (var i = 0; i < pkg.textures.length; i++) {
            var tex_inf = pkg.textures[i];

            // テクスチャのuniform変数の位置
            tex_inf.locTexture = gl.getUniformLocation(pkg.program, tex_inf.name); chk();

            var dim = tex_inf.samplerType == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

            // テクスチャを作る。
            tex_inf.Texture = gl.createTexture(); chk();

            // 指定した位置のテクスチャをアクティブにする。
            gl.activeTexture(this.TEXTUREs[i]); chk();

            // 作成したテクスチャをバインドする。
            gl.bindTexture(dim, tex_inf.Texture); chk();

            if (tex_inf.value instanceof HTMLImageElement || tex_inf.value instanceof HTMLCanvasElement) {
                // テクスチャが画像やキャンバスの場合

                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); chk();
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); chk();    // LINEAR_MIPMAP_NEAREST

                if(tex_inf.value instanceof HTMLCanvasElement){

                }
                else{

                    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true); chk();
                }

                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, tex_inf.value); chk();
                gl.generateMipmap(gl.TEXTURE_2D); chk();
            }
            else {
                // テクスチャが画像やキャンバスでない場合

                gl.texParameteri(dim, gl.TEXTURE_MAG_FILTER, gl.NEAREST); chk();
                gl.texParameteri(dim, gl.TEXTURE_MIN_FILTER, gl.NEAREST); chk();
                gl.texParameteri(dim, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); chk();
                gl.texParameteri(dim, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); chk();
            }
        }
    }

    /*
        テクスチャのデータをセットします。
    */
    setTextureData(pkg: Package) {
        for (var i = 0; i < pkg.textures.length; i++) {
            var tex_inf = pkg.textures[i];

            // テクスチャのuniform変数にテクスチャの番号をセットする。
            gl.uniform1i(tex_inf.locTexture, i); chk();

            var dim = tex_inf.samplerType == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

            // 指定した位置のテクスチャをアクティブにする。
            gl.activeTexture(this.TEXTUREs[i]); chk();

            // テクスチャをバインドする。
            gl.bindTexture(dim, tex_inf.Texture); chk();

            if (tex_inf.value instanceof HTMLImageElement || tex_inf.value instanceof HTMLCanvasElement) {
                // テクスチャが画像の場合

            }
            else {
                // テクスチャが画像でない場合

                var internal_format, format;
                switch (tex_inf.texelType) {
                    case "float":
                        internal_format = gl.R32F;
                        format = gl.RED;
                        break;

                    case "vec2":
                        internal_format = gl.RG32F;
                        format = gl.RG;
                        break;

                    case "vec3":
                        internal_format = gl.RGB32F;
                        format = gl.RGB;
                        break;

                    case "vec4":
                        internal_format = gl.RGBA32F;
                        format = gl.RGBA;
                        break;

                    default:
                        assert(false);
                        break;
                }

                if (dim == gl.TEXTURE_2D) {
                    // 2Dのテクスチャの場合

                    // テクスチャのデータをセットする。
                    gl.texImage2D(gl.TEXTURE_2D, 0, internal_format, tex_inf.shape[1], tex_inf.shape[0], 0, format, gl.FLOAT, tex_inf.value as ArrayBufferView); chk();
                }
                else {
                    // 3Dのテクスチャの場合

                    assert(dim == gl.TEXTURE_3D, "set-Tex");

                    // テクスチャのデータをセットする。
                    gl.texImage3D(gl.TEXTURE_3D, 0, internal_format, tex_inf.shape[2], tex_inf.shape[1], tex_inf.shape[0], 0, format, gl.FLOAT, tex_inf.value as ArrayBufferView); chk();
                }
            }
        }
    }

    makeVertexIndexBuffer(pkg: Package, param: PackageParameter) {
        gl.clearColor(0.0, 0.0, 0.0, 1.0); chk();
        gl.enable(gl.DEPTH_TEST); chk();

        var buf : WebGLBuffer = gl.createBuffer(); chk();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buf); chk();
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, param.VertexIndexBuffer, gl.STATIC_DRAW); chk();

        pkg.vertexIndexBufferInf = {
            value: param.VertexIndexBuffer,
            buffer: buf
        } as VertexIndexBufferInf;
    }

    /*
        ベクトルの次元を返します。
    */
    vecDim(tp: string) {
        if (tp == "vec4") {
            return 4;
        }
        else if (tp == "vec3") {
            return 3;
        }
        else if (tp == "vec2") {
            return 2;
        }
        else {
            return 1;
        }
    }

    /*
        ユニフォーム変数のロケーションをセットします。
    */
    setUniformLocation(pkg: Package) {
        pkg.uniforms.forEach(u => u.locUniform = gl.getUniformLocation(pkg.program, u.name), chk());
    }

    /*
        パッケージを作ります。
    */
    makePackage(param: PackageParameter) {
        var pkg = new Package();
        this.packages[param.id] = pkg;

        pkg.id = param.id;

        if (!param.fragmentShader) {
            // フラグメントシェーダが指定されてない場合

            // デフォルトのフラグメントシェーダをセットする。
            param.fragmentShader = GPGPU.minFragmentShader;
        }

        // シェーダのソースコードを解析する。
        this.parseShader(pkg, param);

        // 頂点シェーダを作る。
        var vertex_shader = this.makeShader(gl.VERTEX_SHADER, param.vertexShader);

        // フラグメントシェーダを作る。
        var fragment_shader = this.makeShader(gl.FRAGMENT_SHADER, param.fragmentShader);

        // プログラムを作る。
        pkg.program = this.makeProgram(vertex_shader, fragment_shader, pkg.varyings);

        // プログラムを使用する。
        gl.useProgram(pkg.program); chk();

        // ユニフォーム変数のロケーションをセットします。
        this.setUniformLocation(pkg);

        // テクスチャを作る。
        this.makeTexture(pkg);

        pkg.attribElementCount = param.elementCount;

        // attribute変数を作る。
        this.makeAttrib(pkg);

        if (pkg.varyings.length != 0) {
            //  varying変数がある場合

            // すべてのvarying変数に対し
            for (let varying of pkg.varyings) {
                var out_buffer_size = this.vecDim(varying.type) * pkg.attribElementCount * Float32Array.BYTES_PER_ELEMENT;

                // Transform Feedbackバッファを作る。
                varying.feedbackBuffer = gl.createBuffer(); chk();

                // バッファをバインドする。
                gl.bindBuffer(gl.ARRAY_BUFFER, varying.feedbackBuffer); chk();
                gl.bufferData(gl.ARRAY_BUFFER, out_buffer_size, gl.STATIC_COPY); chk();
                gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
            }

            // Transform Feedbackを作る。
            pkg.transformFeedback = gl.createTransformFeedback(); chk();
        }

        if (param.VertexIndexBuffer) {
            this.makeVertexIndexBuffer(pkg, param);
        }

        return pkg;
    }

    /*
        attribute変数のデータをセットします。
    */
    setAttribData(pkg: Package) {
        // すべてのattribute変数に対し
        for (let attrib of pkg.attributes) {
            var dim = this.vecDim(attrib.type);

            gl.bindBuffer(gl.ARRAY_BUFFER, attrib.AttribBuffer); chk();

            // 指定した位置のattribute変数の要素数(dim)と型(float)をセットする。
            gl.vertexAttribPointer(attrib.AttribLoc, dim, gl.FLOAT, false, 0, 0); chk();

            // attribute変数のデータをセットする。
            gl.bufferData(gl.ARRAY_BUFFER, attrib.value, gl.STATIC_DRAW);
        }
    }

    /*
        uniform変数のデータをセットします。
    */
    setUniformsData(pkg: Package) {
        // すべてのuniform変数に対し
        for (let u of pkg.uniforms) {
            if (u.value instanceof Float32Array) {
                // 値が配列の場合

                switch (u.type) {
                    case "mat4":
                        gl.uniformMatrix4fv(u.locUniform, false, u.value); chk();
                        break;
                    case "mat3":
                        gl.uniformMatrix3fv(u.locUniform, false, u.value); chk();
                        break;
                    case "vec4":
                        gl.uniform4fv(u.locUniform, u.value); chk();
                        break;
                    case "vec3":
                        gl.uniform3fv(u.locUniform, u.value); chk();
                        break;
                    case "vec2":
                        gl.uniform2fv(u.locUniform, u.value); chk();
                        break;
                    case "float":
                        gl.uniform1fv(u.locUniform, u.value); chk();
                        break;
                    default:
                        assert(false);
                        break;
                }
            }
            else {
                // 値が配列でない場合

                if (u.type == "int" || u.type == "bool") {

                    gl.uniform1i(u.locUniform, u.value); chk();
                }
                else {

                    gl.uniform1f(u.locUniform, u.value); chk();
                }
            }
        }
    }

    /*
        パラメータの引数の値をコピーします。
    */
    copyParamArgsValue(param: PackageParameter, pkg: Package){
        for(let args of[ pkg.attributes, pkg.uniforms, pkg.textures, pkg.varyings ]) {
            for (let arg of args) {
                var val = param.args[arg.name];
                assert(val != undefined);
                if (args == pkg.textures) {
                    // テクスチャ情報の場合

                    arg.value = val.value;
                }
                else {
                    // テクスチャでない場合

                    arg.value = val;
                }
            }
        }
    }

    /*
        計算します。
    */
    compute(param: PackageParameter, drawable: Drawable = undefined) {
        var pkg = this.packages[param.id] as Package;
        if (!pkg) {
            // パッケージが未作成の場合

            // パッケージを作る。
            pkg = this.makePackage(param);
        }
        else {

            gl.useProgram(pkg.program); chk();
        }

        // 実引数の値をコピーする。
        this.copyParamArgsValue(param, pkg);

        // attribute変数の値をセットする。
        this.setAttribData(pkg);

        gl.useProgram(pkg.program); chk();

        // テクスチャの値のセットする。
        this.setTextureData(pkg);

        // ユニフォーム変数の値をセットする。
        this.setUniformsData(pkg);

        if (pkg.varyings.length == 0) {
            //  描画する場合

            gl.viewport(0, 0, this.canvas.width, this.canvas.height); chk();

            // 頂点インデックスバッファをバインドする。
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, pkg.vertexIndexBufferInf.buffer); chk();

            if(drawable instanceof Points){

                // 頂点のリストを描画する。
                gl.drawElements(gl.POINTS, pkg.vertexIndexBufferInf.value.length, gl.UNSIGNED_SHORT, 0); chk();
            }
            else if(drawable instanceof Lines){

                // 線分のリストを描画する。
                gl.drawElements(gl.LINES, pkg.vertexIndexBufferInf.value.length, gl.UNSIGNED_SHORT, 0); chk();
            }
            else{

                // 三角形のリストを描画する。
                gl.drawElements(gl.TRIANGLES, pkg.vertexIndexBufferInf.value.length, gl.UNSIGNED_SHORT, 0); chk();
            }
        }
        else {
            //  描画しない場合

            // ラスタライザを無効にする。
            gl.enable(gl.RASTERIZER_DISCARD); chk();

            // Transform Feedbackをバインドする。
            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, pkg.transformFeedback); chk();

            // すべてのvarying変数に対し
            for (var i = 0; i < pkg.varyings.length; i++) {
                var varying = pkg.varyings[i];

                // Transform Feedbackのバッファをバインドする。
                gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, varying.feedbackBuffer); chk();
            }

            // Transform Feedbackを開始する。
            gl.beginTransformFeedback(gl.POINTS); chk();    // TRIANGLES

            // 点ごとの描画をする。
            gl.drawArrays(gl.POINTS, 0, pkg.attribElementCount); chk();

            // Transform Feedbackを終了する。
            gl.endTransformFeedback(); chk();

            // ラスタライザを有効にする。
            gl.disable(gl.RASTERIZER_DISCARD); chk();

            // すべてのvarying変数に対し
            for (var i = 0; i < pkg.varyings.length; i++) {
                varying = pkg.varyings[i];

                // Transform Feedbackのバッファのバインドを解く。
                gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, null); chk();

                // ARRAY_BUFFERにバインドする。
                gl.bindBuffer(gl.ARRAY_BUFFER, varying.feedbackBuffer); chk();

                // ARRAY_BUFFERのデータを取り出す。
                gl.getBufferSubData(gl.ARRAY_BUFFER, 0, varying.value); chk();

                // ARRAY_BUFFERのバインドを解く。
                gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
            }

            // Transform Feedbackのバインドを解く。
            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null); chk();
        }

        // プログラムの使用を終了する。
        gl.useProgram(null); chk();
    }

    /*
        標準のシェーダの文字列をセットします。
    */
    setStandardShaderString() {
    }

    draw(drawable: Drawable, worldMat: Float32Array, viewMat: Float32Array, projMat: Float32Array){

        let modelMat = mat4.create();
        mat4.multiply(worldMat, drawable.transform, modelMat);

        let viewModelMat = mat4.create();
        mat4.multiply(viewMat, modelMat, viewModelMat);

        if(drawable instanceof ComponentDrawable){
            for(let child of drawable.children){
                this.draw(child, modelMat, viewMat, projMat)
            }
        }

        let param = drawable.getParam();

        if(param != null){

            let projViewModelMat = mat4.create();
            mat4.multiply(projMat, viewModelMat, projViewModelMat);

            let normalMatrix = mat3.create();
            mat4.toInverseMat3(viewMat, normalMatrix);
            mat3.transpose(normalMatrix);


            param.args["uPMVMatrix"] = projViewModelMat;
            param.args["uNMatrix"] = normalMatrix;

            this.compute(param, drawable);
        }
    }

    /*
        3D表示をします。
    */
    drawScene() {
        // カラーバッファと深度バッファをクリアする。
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); chk();

        let viewMat = mat4.create();   // drawable.transform
        mat4.identity(viewMat);

        mat4.translate(viewMat, [this.drawParam.x, this.drawParam.y, this.drawParam.z]);

        mat4.rotate(viewMat, this.drawParam.xRot, [1, 0, 0]);
        mat4.rotate(viewMat, this.drawParam.yRot, [0, 1, 0]);

        let projMat = mat4.create();
        mat4.perspective(45, this.canvas.width / this.canvas.height, 0.1, 100.0, projMat);

        for(let drawable of this.drawables){

            let worldMat = mat4.create();
            mat4.identity(worldMat);
            this.draw(drawable, worldMat, viewMat, projMat);
        }

        // 次の再描画でdrawSceneが呼ばれるようにする。
        window.requestAnimationFrame(this.drawScene.bind(this));
    }

    /*
        3D表示を開始します。
    */
    startDraw3D(drawables: Drawable[]) {
        this.drawables = drawables;
        this.drawParam = {
            xRot : 0,
            yRot : 0,
            x    : 0,
            y    : 0,
            z    : -5.0
        } as DrawParam;

        var lastMouseX = null;
        var lastMouseY = null;

        // mousemoveのイベント リスナーを登録する。
        this.canvas.addEventListener('mousemove', (ev: MouseEvent)=> {
            var newX = ev.clientX;
            var newY = ev.clientY;

            if (ev.buttons != 0 && lastMouseX != null) {

                if(ev.shiftKey){

                    this.drawParam.x += (newX - lastMouseX) / 300;
                    this.drawParam.y -= (newY - lastMouseY) / 300;
                }
                else{

                    this.drawParam.xRot += (newY -lastMouseY) / 300;
                    this.drawParam.yRot += (newX - lastMouseX) / 300;
                }
            }

            lastMouseX = newX
            lastMouseY = newY;
        });

        // touchmoveのイベント リスナーを登録する。
        this.canvas.addEventListener('touchmove', (ev: TouchEvent)=> {
            // タッチによる画面スクロールを止める
            ev.preventDefault(); 

            var newX = ev.changedTouches[0].clientX;
            var newY = ev.changedTouches[0].clientY;

            if (lastMouseX != null) {

                this.drawParam.xRot += (newY - lastMouseY) / 300;
                this.drawParam.yRot += (newX - lastMouseX) / 300;
            }

            lastMouseX = newX
            lastMouseY = newY;
        }, false);

        // wheelのイベント リスナーを登録する。
        this.canvas.addEventListener("wheel",  (ev: WheelEvent)=> {
            this.drawParam.z += 0.002 * ev.deltaY;

            // ホイール操作によるスクロールを無効化する
            ev.preventDefault();
        });

        // 3D表示をする。
        this.drawScene();
    }
}

}

/*
    GPGPUのオブジェクトを作ります。

    この関数の内部に関数やクラスを入れて外部から参照されないようにします。
*/
function CreateGPGPU(canvas: HTMLCanvasElement) {
    return new gpgpu.GPGPU(canvas);
}