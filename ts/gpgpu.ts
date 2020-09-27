namespace gpgputs {

declare namespace mat4 {
    function create() : Float32Array;
    function translate(out: Float32Array, a: Float32Array, v: [number, number, number]) : void;
    function rotateX(out : Float32Array, a : Float32Array, rad : number) : Float32Array;
    function rotateY(out : Float32Array, a : Float32Array, rad : number) : Float32Array;
    function rotateZ(out : Float32Array, a : Float32Array, rad : number) : Float32Array;
    function scale(out : Float32Array, a : Float32Array, v : [number, number, number]) : Float32Array;
    function multiply(out : Float32Array, a : Float32Array, b : Float32Array) : Float32Array;
    function perspective(out : Float32Array, fovy : number, aspect : number, near : number, far : number) : Float32Array;
}

declare namespace mat3 {
    function create() : Float32Array;
    function normalFromMat4(out : Float32Array, a : Float32Array) : Float32Array;
}

type TextureValue = Float32Array | HTMLImageElement | HTMLCanvasElement;
export type Mesh = any;

export let gl : WebGL2RenderingContext;

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
export function chk() {
    function chk() {
        let sts = gl.getError();
        if(sts != gl.NO_ERROR){
    
            assert(false, `chk:${sts.toString(16)}`);
        }
    }
}


    /*
        ベクトルの次元を返します。
    */
export function vecDim(tp: string) : number {
    if (tp == "vec4") {
        return 4;
    }
    else if (tp == "vec3") {
        return 3;
    }
    else if (tp == "vec2") {
        return 2;
    }
    else if(tp == "float"){
        return 1;
    }
    else if(tp == "mat3"){
        return 3 * 3;
    }
    else if(tp == "mat4"){
        return 4 * 4;
    }
    else{
        throw new Error();
    }
}

export function getDrawMode(mode: string) : number{
    switch(mode){
    case "POINTS"        : return gl.POINTS;
    case "LINES"         : return gl.LINES;
    case "TRIANGLES"     : return gl.TRIANGLES;
    case "TRIANGLE_FAN"  : return gl.TRIANGLE_FAN;
    case "TRIANGLE_STRIP": return gl.TRIANGLE_STRIP;
    }

    throw new Error();
}


export function getDrawModeText(mode: number) : string{
    switch(mode){
    case gl.POINTS        : return "POINTS";
    case gl.LINES         : return "LINES";
    case gl.TRIANGLES     : return "TRIANGLES";
    case gl.TRIANGLE_FAN  : return "TRIANGLE_FAN";
    case gl.TRIANGLE_STRIP: return "TRIANGLE_STRIP";
    }

    throw new Error();
}

export class Vec3 {
    x: number;
    y: number;
    z: number;

    constructor(x: number, y: number, z: number){
        this.x = x;
        this.y = y;
        this.z = z;
    }

    len(){
        return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
    }

    unit() : Vec3 {
        const len = this.len();

        if(len == 0){
            return new Vec3(0, 0, 0);
        }
        else{
            return new Vec3(this.x / len, this.y / len, this.z / len);
        }
    }

    mul(n: number){
        return new Vec3(n * this.x, n * this.y, n * this.z);
    }

    add(v: Vec3) {
        return new Vec3( this.x + v.x, this.y + v.y, this.z + v.z );
    }

    sub(v: Vec3) {
        return new Vec3( this.x - v.x, this.y - v.y, this.z - v.z );
    }
    
    dot(v: Vec3) {
        return this.x * v.x + this.y * v.y + this.z * v.z;
    }
    
    cross(v: Vec3) {
        return new Vec3(this.y * v.z - this.z * v.y, this.z * v.x - this.x * v.z, this.x * v.y - this.y * v.x);
    }
}

export class Vertex extends Vec3 {
    nx: number = 0;
    ny: number = 0;
    nz: number = 0;
    texX: number = 0;
    texY: number = 0;

    adjacentVertexes: Vertex[];

    constructor(x: number, y: number, z: number) {
        super(x, y, z);
        this.adjacentVertexes = [];
    }
}


export class Color {
    r: number;
    g: number;
    b: number;
    a: number;

    constructor(r:number, g: number, b: number, a: number){
        this.r = r;
        this.g = g;
        this.b = b;
        this.a = a;
    }

    static get red(): Color {
        return new Color(1, 0, 0, 1);
    }

    static get green(): Color {
        return new Color(0, 1, 0, 1);
    }

    static get blue(): Color {
        return new Color(0, 0, 1, 1);
    }
}

export class AbsDrawable {
    transform: Float32Array;

    constructor(){
        this.transform = mat4.create();
    }

    move(x: number, y: number, z: number) : AbsDrawable {
        mat4.translate(this.transform, this.transform, [x, y, z]);

        return this;
    }

    scale(x: number, y: number, z: number) : AbsDrawable {
        mat4.scale(this.transform, this.transform, [x, y, z]);

        return this;
    }

    clear(){
        throw new Error();
    }
}

export class Package extends AbsDrawable{
    static count = 0;

    id!: string;
    numInput: number | undefined;
    mode!: GLenum;
    vertexShader!: string;
    fragmentShader: string = GPGPU.minFragmentShader;

    args: Mesh;
    VertexIndexBuffer: Uint16Array | Uint32Array | undefined;

    program: WebGLProgram | undefined;
    transformFeedback: WebGLTransformFeedback | undefined;

    vertexIndexBufferInf: WebGLBuffer | undefined;
    textures: TextureInfo[] = [];
    numGroup: number | undefined;
    varyings: ArgInf[] = [];
    attributes: ArgInf[] = [];
    uniforms: ArgInf[] = [];
    pipes: BindArg[] = [];
    update : (()=>void) | undefined = undefined;
    fps: number = 0;
    time: number = 0;

    constructor(obj: any = undefined){
        super();
        if(obj != undefined){
            Object.assign(this, obj);
        }
        assert(this.mode != undefined);

        if(this.id == undefined){
            this.id = `${this.constructor.name}_${Package.count++}`;
        }
    }

    bind(outName: string, inName: string, inPackage: Package | undefined = undefined){
        this.pipes.push(new BindArg(outName, inName, this, (inPackage == undefined ? this : inPackage)));
    }

    /*
        指定したidのWebGLのオブジェクトをすべて削除します。
    */
    clear() {
        // 指定したidのパッケージがある場合

        gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();

        // テクスチャのバインドを解く。
        gl.bindTexture(gl.TEXTURE_2D, null); chk();
        gl.bindTexture(gl.TEXTURE_3D, null); chk();

        if (this.vertexIndexBufferInf) {

            // バッファを削除する。
            gl.deleteBuffer(this.vertexIndexBufferInf); chk();
        }

        // すべてのvarying変数に対し
        for (let varying of this.varyings) {
            if (varying.feedbackBuffer) {
                // Transform Feedbackバッファがある場合

                // バッファを削除する。
                gl.deleteBuffer(varying.feedbackBuffer); chk();
            }
        }

        if (this.transformFeedback) {
            // Transform Feedbackがある場合

            gl.deleteTransformFeedback(this.transformFeedback); chk();
        }

        // すべてのテクスチャを削除する。
        this.textures.forEach(x => gl.deleteTexture(x.Texture!), chk())

        // プログラムを削除する。
        gl.deleteProgram(this.program!); chk();
    }
    
    ready() {
    }
}

export class Drawable extends Package {
    constructor(obj: any = undefined){
        super(obj);
    }

    getVertexColors(color: Color, numVertex: number) : number[] {
        let vertexColors: number[] = [];

        range(numVertex).forEach(x => {
            vertexColors.push(color.r, color.g, color.b, color.a);
        });

        return vertexColors;
    }
}


export class Points extends Drawable {
    constructor(points: Float32Array, colors: Float32Array, pointSize: number){
        super({
            mode: gl.POINTS,
            vertexShader: VertexShader.points,
            fragmentShader: GPGPU.pointFragmentShader,
            args: {
                vertexPosition: points,
                vertexColor: colors,
                pointSize  : pointSize
            } as any as Mesh,
            VertexIndexBuffer: new Uint16Array(range(points.length / 3))
        });

        this.id = `${this.constructor.name}.${Package.count++}`;
    }

    makeVertexPosition(vertices: Vertex[]) : Float32Array {
        const positions = new Float32Array(3 * vertices.length);

        let base = 0;
        for(let i = 0; i < vertices.length; i++){
            let p = vertices[i];
            positions[base    ] = p.x;
            positions[base + 1] = p.y;
            positions[base + 2] = p.z;

            base += 3;
        }

        return positions;
    }

    updateNotUsed(points: Float32Array, color: Color, pointSize: number){
        // 色の配列
        let vertexColors = this.getVertexColors(color, points.length / 3);
    
        let mesh = this.args as Mesh;
        mesh.vertexPosition = points;
        mesh.vertexColor = new Float32Array(vertexColors);
        mesh.pointSize = pointSize;

        this.VertexIndexBuffer = new Uint16Array(range(points.length / 3));
    }
}

export class Lines extends Drawable {
    constructor(vertices: Vertex[], color: Color){
        super({
            mode: gl.LINES,
            vertexShader: VertexShader.lines,
            fragmentShader: GPGPU.pointFragmentShader,
            VertexIndexBuffer: new Uint16Array(range(vertices.length))
        });

        this.id = `${this.constructor.name}.${Package.count++}`;
        
        // 色の配列
        let vertexColors = this.getVertexColors(color, vertices.length);
    
        const positions : number[] = [];
        vertices.forEach(p => positions.push(p.x, p.y, p.z));

        this.args = {
            vertexPosition: new Float32Array(positions),
            vertexColor: new Float32Array(vertexColors),
        } as any as Mesh;
    }
}


export class ComponentDrawable extends AbsDrawable {
    children: AbsDrawable[];

    constructor(children: AbsDrawable[]){
        super();
        this.children = children.slice();
    }

    clear(){
        for(let pkg of this.children){
            pkg.clear();
        }
    }
}

export class UserDef extends Drawable {
    constructor(mode: GLenum,  vertexShader: string, fragmentShader: string, args: any = {}){
        super({
            mode: mode,
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            args: args
        });
    }
}

export class UserMesh extends UserDef {
    constructor(mode: GLenum, vertexShader: string, fragmentShader: string, numInput: number, numGroup: number | undefined = undefined){
        super(mode, vertexShader, fragmentShader);
        this.numInput = numInput;
        this.numGroup  = numGroup;
    }
}



/*
    テクスチャ情報

    テクスチャのテクセルの型、サイズ、値の情報を持ちます。
*/
export class TextureInfo {
    dirty: boolean = true;
    name: string | undefined;
    isArray: boolean | undefined;
    texelType: string | null;
    samplerType: string | undefined;
    shape: number[] | null;
    value: TextureValue | undefined;

    Texture: WebGLTexture | undefined;
    locTexture: WebGLUniformLocation | undefined;

    /*
        TextureInfoのコンストラクタ
    */
    constructor(texel_type: string | null, shape: number[] | null, value: TextureValue | undefined = undefined) {
        assert(texel_type != null || value != undefined && !(value instanceof Float32Array));
        if(shape != null){
            assert(shape.length == 2 || shape.length == 3);
        }

        if(value == undefined){
            let dim = vecDim(texel_type!);

            let cnt = shape!.reduce((x,y) => x*y, 1);
            value = new Float32Array(dim * cnt);
        }

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
    value: Float32Array | number;
    type: string;
    isArray: boolean;

    feedbackBuffer: WebGLBuffer | undefined;
    locUniform: WebGLUniformLocation | undefined;
    AttribBuffer: WebGLBuffer | undefined;
    AttribLoc: number | undefined;

    constructor(name: string, value: Float32Array | number, type: string, is_array: boolean){
        this.name = name;
        this.value = value;
        this.type  = type;
        this.isArray = is_array;
    }
}

class BindArg{
    outName   : string;
    inName   : string;
    outPackage: Package;
    inPackage: Package;

    constructor(outName   : string, inName   : string, outPackage: Package, inPackage: Package){
        let out_val = outPackage.args[outName] as Float32Array;
        assert(out_val instanceof Float32Array);
        let inArg = inPackage.args[inName];

        if(inArg instanceof Float32Array){

            assert(out_val.length == inArg.length);
        }
        else if(inArg instanceof TextureInfo && inArg.value instanceof Float32Array){

            assert(out_val.length == inArg.value.length);
        }
        else{
            throw new Error();
        }

        this.outName    = outName;
        this.inName     = inName;

        this.outPackage = outPackage;
        this.inPackage  = inPackage;
    }
}

export class DrawParam{
    xRot : number;
    yRot : number;
    zRot : number;
    x    : number;
    y    : number;
    z    : number;

    constructor(xrot: number, yrot: number, zrot: number, x: number, y: number, z: number){
        this.xRot = xrot;
        this.yRot = yrot;
        this.zRot = zrot;
        this.x = x;
        this.y = y;
        this.z = z;
    }
}

export class UI3D {
    lastMouseX : number | null = null;
    lastMouseY : number | null = null;

    mousedown : ((ev: MouseEvent, drawParam: DrawParam)=>void) | undefined;

    mouseup : ((ev: MouseEvent, drawParam: DrawParam)=>void) | undefined;

    mousemove : ((ev: MouseEvent, drawParam: DrawParam)=>void) | undefined;

    pointerdown : ((ev: PointerEvent, drawParam: DrawParam)=>void) | undefined;
    pointerup   : ((ev: PointerEvent, drawParam: DrawParam)=>void) | undefined;

    pointermove = (ev: PointerEvent, drawParam: DrawParam)=>{
        var newX = ev.clientX;
        var newY = ev.clientY;

        if (ev.buttons != 0 && this.lastMouseX != null && this.lastMouseY != null) {

            if(ev.shiftKey){

                drawParam.x += (newX - this.lastMouseX) / 300;
                drawParam.y -= (newY - this.lastMouseY) / 300;
            }
            else{

                drawParam.xRot += (newY - this.lastMouseY) / 100;
                drawParam.yRot += (newX - this.lastMouseX) / 100;
            }
        }

        this.lastMouseX = newX
        this.lastMouseY = newY;
    }

    touchmove = (ev: TouchEvent, drawParam: DrawParam)=>{
        // タッチによる画面スクロールを止める
        ev.preventDefault(); 

        var newX = ev.changedTouches[0].clientX;
        var newY = ev.changedTouches[0].clientY;

        if (this.lastMouseX != null && this.lastMouseY != null) {

            drawParam.xRot += (newY - this.lastMouseY) / 300;
            drawParam.yRot += (newX - this.lastMouseX) / 300;
        }

        this.lastMouseX = newX
        this.lastMouseY = newY;
    }


    wheel = (ev: WheelEvent, drawParam: DrawParam)=>{
        drawParam.z += 0.002 * ev.deltaY;

        // ホイール操作によるスクロールを無効化する
        ev.preventDefault();
    }
}

export interface DrawScenelistener {
    beforeDraw() : void;
    afterDraw(projViewMat: Float32Array)  : void;
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
    static readonly minFragmentShader = `
out vec4 color;

void main(){
    color = vec4(1.0);
}
`;

    // デフォルトの動作のフラグメントシェーダ
    static readonly defaultFragmentShader = `
in vec3 vLightWeighting;
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

    static readonly planeTextureFragmentShader = `
in vec3 vLightWeighting;
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

    static readonly pointFragmentShader = `
in  vec4 fragmentColor;
out vec4 color;

void main(void) {
    color = fragmentColor;
}
`;

    static readonly vertexPositionShader = `
in vec3 in_position;
        
void main(void) {
    gl_Position = vec4(in_position, 1.0);
}`;

    canvas: HTMLCanvasElement;
    TEXTUREs: number[];
    packages: Package[] = [];
    drawables: AbsDrawable[] = [];
    drawParam: DrawParam = new DrawParam(0, 0, 0, 0, 0, -5.0);
    ui3D : UI3D;
    drawScenelistener : DrawScenelistener | null = null;

    /*
        GPGPUのコンストラクタ
    */
    constructor(canvas: HTMLCanvasElement | undefined, ui3d: UI3D) {
        console.log("init WebGL");

        this.ui3D = ui3d;

        if (canvas == undefined) {
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

        const ext = gl.getExtension("EXT_color_buffer_float");
        if (!ext) {
            throw new Error("need EXT_color_buffer_float");
        }

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
    makeTextureInfo(texel_type: string, shape: number[], value: TextureValue | undefined = undefined) {
        return new TextureInfo(texel_type, shape, value);
    }

    /*
        WebGLのオブジェクトをすべて削除します。
    */
    clearAll() {
        for(let pkg of this.packages){
            pkg.clear();
        }

        this.packages = [];

        while(this.drawables.length != 0){
            this.drawables.pop()!.clear();
        }
    }

    /*
        シェーダのソースコードを解析します。
    */
    parseShader(pkg: Package) {
        let fragIn : string[] = [];

        // 頂点シェーダとフラグメントシェーダのソースに対し
        for(let shader_text of[ pkg.fragmentShader, pkg.vertexShader ]) {
            if(shader_text == GPGPU.vertexPositionShader){
                continue;
            }

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
                var type_name = tokens[1];
                var tkn2 = tokens[2];

                if (tkn0 != "in" && tkn0 != "uniform" && tkn0 != "out") {
                    // 最初のトークンが in, uniform, out でない場合
                    continue;
                }

                assert(type_name == "int" || type_name == "float" || type_name == "vec2" || type_name == "vec3" || type_name == "vec4" ||
                    type_name == "sampler2D" || type_name == "sampler3D" ||
                    type_name == "mat4" || type_name == "mat3" || type_name == "bool");

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

                if (shader_text == pkg.fragmentShader && tkn0 != "uniform") {
                    // フラグメントシェーダで uniform でない場合 ( フラグメントシェーダの入力(in)と出力(out)はアプリ側では使わない。 )

                    if(tkn0 == "in"){
                        fragIn.push(arg_name);
                    }

                    continue;
                }

                // 変数の値を得る。
                var arg_val = pkg.args[arg_name];

                if (arg_val == undefined) {
                    
                    if(tokens[0] == "out"){
                        if(fragIn.includes(arg_name)){

                            continue;
                        }

                        let dim = vecDim(type_name);

                        let value = new Float32Array(dim * pkg.numInput!);
                        pkg.args[arg_name] = value;
                        let arg_inf = new ArgInf(arg_name, value, type_name, is_array );
                        pkg.varyings.push(arg_inf);
                        continue;
                    }
                }

                if (type_name == "sampler2D" || type_name == "sampler3D") {
                    // テクスチャのsamplerの場合

                    assert(tokens[0] == "uniform" && arg_val instanceof TextureInfo);
                    let tex_inf = arg_val as TextureInfo;

                    // 変数名をセットする。
                    tex_inf.name = arg_name;

                    // samplerのタイプをセットする。
                    tex_inf.samplerType = type_name;

                    // 配列かどうかをセットする。
                    tex_inf.isArray = is_array;

                    // テクスチャの配列に追加する。
                    pkg.textures.push(tex_inf);
                }
                else {
                    // テクスチャのsamplerでない場合

                    // 変数の名前、値、型、配列かどうかをセットする。
                    var arg_inf = new ArgInf(arg_name, arg_val as (Float32Array | number), type_name, is_array );

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
        var prg = gl.createProgram()!; chk();

        // 頂点シェーダをアタッチする。
        gl.attachShader(prg, vertex_shader); chk();

        // フラグメントシェーダをアタッチする。
        gl.attachShader(prg, fragment_shader); chk();

        if (varyings.length != 0) {
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
        var shader = gl.createShader(type)!; chk();

        // シェーダにソースをセットする。
        gl.shaderSource(shader, source); chk();

        // シェーダをコンパイルする。
        gl.compileShader(shader); chk();

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            // コンパイル エラーの場合

            const s = gl.getShaderInfoLog(shader);
            const src = Array.from(source.split('\n').entries()).map(x => `${x[0] + 1} ${x[1]}`).join('\n');
            console.log(s + "\n" + '-'.repeat(40) + "\n" + src);
            alert(s);
            throw new Error();
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
            var attrib_dim = vecDim(attrib.type);

            // 要素の個数
            var elemen_count = (attrib.value as Float32Array).length / attrib_dim;

            if(pkg.numInput == undefined){

                pkg.numInput = elemen_count;
            }
            else{
                assert(pkg.numInput == elemen_count);
            }

            // バッファを作る。
            attrib.AttribBuffer = gl.createBuffer()!; chk();

            // attribute変数の位置
            attrib.AttribLoc = gl.getAttribLocation(pkg.program!, attrib.name); chk();

            // 指定した位置のattribute配列を有効にする。
            gl.enableVertexAttribArray(attrib.AttribLoc); chk();

            // attribute変数の位置と変数名をバインドする。
            gl.bindAttribLocation(pkg.program!, attrib.AttribLoc, attrib.name);
        }

        assert(pkg.numInput != undefined);
    }

    /*
        テクスチャを作ります。
    */
    makeTexture(pkg: Package) {
        // すべてのテクスチャに対し
        for (var i = 0; i < pkg.textures.length; i++) {
            var tex_inf = pkg.textures[i];

            // テクスチャのuniform変数の位置
            tex_inf.locTexture = gl.getUniformLocation(pkg.program!, tex_inf.name!)!; chk();

            var dim = tex_inf.samplerType == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

            // テクスチャを作る。
            tex_inf.Texture = gl.createTexture()!; chk();

            // 指定した位置のテクスチャをアクティブにする。
            gl.activeTexture(this.TEXTUREs[i]); chk();

            // 作成したテクスチャをバインドする。
            gl.bindTexture(dim, tex_inf.Texture); chk();

            if (tex_inf.value instanceof HTMLImageElement || tex_inf.value instanceof HTMLCanvasElement) {
                // テクスチャが画像やキャンバスの場合

                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); chk();
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); chk();    // LINEAR_MIPMAP_NEAREST
                gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true); chk();
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
            gl.uniform1i(tex_inf.locTexture!, i); chk();

            var dim = tex_inf.samplerType == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

            // 指定した位置のテクスチャをアクティブにする。
            gl.activeTexture(this.TEXTUREs[i]); chk();

            // テクスチャをバインドする。
            gl.bindTexture(dim, tex_inf.Texture!); chk();

            if(! tex_inf.dirty){
                continue;
            }
            tex_inf.dirty = false;

            if (tex_inf.value instanceof HTMLImageElement || tex_inf.value instanceof HTMLCanvasElement) {
                // テクスチャが画像の場合

                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, tex_inf.value); chk();
                gl.generateMipmap(gl.TEXTURE_2D); chk();
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
                        throw new Error();
                }

                if (dim == gl.TEXTURE_2D) {
                    // 2Dのテクスチャの場合

                    // テクスチャのデータをセットする。
                    gl.texImage2D(gl.TEXTURE_2D, 0, internal_format, tex_inf.shape![1], tex_inf.shape![0], 0, format, gl.FLOAT, tex_inf.value as ArrayBufferView); chk();
                }
                else {
                    // 3Dのテクスチャの場合

                    assert(dim == gl.TEXTURE_3D, "set-Tex");

                    // テクスチャのデータをセットする。
                    gl.texImage3D(gl.TEXTURE_3D, 0, internal_format, tex_inf.shape![2], tex_inf.shape![1], tex_inf.shape![0], 0, format, gl.FLOAT, tex_inf.value as ArrayBufferView); chk();
                }
            }
        }
    }

    makeVertexIndexBuffer(pkg: Package) {
        gl.clearColor(0.0, 0.0, 0.0, 1.0); chk();
        gl.enable(gl.DEPTH_TEST); chk();

        pkg.vertexIndexBufferInf = gl.createBuffer()!; chk();

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, pkg.vertexIndexBufferInf); chk();
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, pkg.VertexIndexBuffer!, gl.STATIC_DRAW); chk();
    }

    /*
        ユニフォーム変数のロケーションをセットします。
    */
    setUniformLocation(pkg: Package) {
        pkg.uniforms.forEach(u => u.locUniform = gl.getUniformLocation(pkg.program!, u.name)!, chk());
    }

    /*
        パッケージを作ります。
    */
    makePackage(pkg: Package) {
        // シェーダのソースコードを解析する。
        this.parseShader(pkg);

        // 頂点シェーダを作る。
        var vertex_shader = this.makeShader(gl.VERTEX_SHADER, pkg.vertexShader);

        // フラグメントシェーダを作る。
        var fragment_shader = this.makeShader(gl.FRAGMENT_SHADER, pkg.fragmentShader);

        // プログラムを作る。
        pkg.program = this.makeProgram(vertex_shader, fragment_shader, pkg.varyings);

        // プログラムを使用する。
        gl.useProgram(pkg.program); chk();

        // ユニフォーム変数のロケーションをセットします。
        this.setUniformLocation(pkg);

        // テクスチャを作る。
        this.makeTexture(pkg);

        // attribute変数を作る。
        this.makeAttrib(pkg);

        if (pkg.varyings.length != 0) {
            //  varying変数がある場合

            // すべてのvarying変数に対し
            for (let varying of pkg.varyings) {
                var out_buffer_size = vecDim(varying.type) * pkg.numInput! * Float32Array.BYTES_PER_ELEMENT;

                // Transform Feedbackバッファを作る。
                varying.feedbackBuffer = gl.createBuffer()!; chk();

                // バッファをバインドする。
                gl.bindBuffer(gl.ARRAY_BUFFER, varying.feedbackBuffer); chk();
                gl.bufferData(gl.ARRAY_BUFFER, out_buffer_size, gl.STATIC_COPY); chk();
                gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
            }

            // Transform Feedbackを作る。
            pkg.transformFeedback = gl.createTransformFeedback()!; chk();
        }

        if (pkg.VertexIndexBuffer) {
            this.makeVertexIndexBuffer(pkg);
        }
    }

    /*
        attribute変数のデータをセットします。
    */
    setAttribData(pkg: Package) {
        // すべてのattribute変数に対し
        for (let attrib of pkg.attributes) {
            var dim = vecDim(attrib.type);

            gl.bindBuffer(gl.ARRAY_BUFFER, attrib.AttribBuffer!); chk();

            // 指定した位置のattribute変数の要素数(dim)と型(float)をセットする。
            gl.vertexAttribPointer(attrib.AttribLoc!, dim, gl.FLOAT, false, 0, 0); chk();

            // attribute変数のデータをセットする。
            gl.bufferData(gl.ARRAY_BUFFER, attrib.value as Float32Array, gl.STATIC_DRAW);
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
                        gl.uniformMatrix4fv(u.locUniform!, false, u.value); chk();
                        break;
                    case "mat3":
                        gl.uniformMatrix3fv(u.locUniform!, false, u.value); chk();
                        break;
                    case "vec4":
                        gl.uniform4fv(u.locUniform!, u.value); chk();
                        break;
                    case "vec3":
                        gl.uniform3fv(u.locUniform!, u.value); chk();
                        break;
                    case "vec2":
                        gl.uniform2fv(u.locUniform!, u.value); chk();
                        break;
                    case "float":
                        gl.uniform1fv(u.locUniform!, u.value); chk();
                        break;
                    default:
                        assert(false);
                        break;
                }
            }
            else {
                // 値が配列でない場合

                if (u.type == "int" || u.type == "bool") {

                    gl.uniform1i(u.locUniform!, u.value); chk();
                }
                else {

                    gl.uniform1f(u.locUniform!, u.value); chk();
                }
            }
        }
    }

    /*
        パラメータの引数の値をコピーします。
    */
    copyParamArgsValue(pkg: Package){
        for(let args of[ pkg.attributes, pkg.uniforms, pkg.textures, pkg.varyings ]) {
            for (let arg of args) {
                var val = pkg.args[arg.name!];
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
    compute(pkg: Package) {
        if (pkg.program == undefined) {
            // パッケージが未作成の場合

            // パッケージを作る。
            this.makePackage(pkg);
        }
        else {

            gl.useProgram(pkg.program); chk();
        }

        // 実引数の値をコピーする。
        this.copyParamArgsValue(pkg);

        // attribute変数の値をセットする。
        this.setAttribData(pkg);

        gl.useProgram(pkg.program!); chk();

        // テクスチャの値をセットする。
        this.setTextureData(pkg);

        // ユニフォーム変数の値をセットする。
        this.setUniformsData(pkg);

        if(pkg.fragmentShader == GPGPU.minFragmentShader){

            // ラスタライザを無効にする。
            gl.enable(gl.RASTERIZER_DISCARD); chk();
        }

        if (pkg.varyings.length != 0){

            // Transform Feedbackをバインドする。
            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, pkg.transformFeedback!); chk();

            // すべてのvarying変数に対し
            for (var i = 0; i < pkg.varyings.length; i++) {
                var varying = pkg.varyings[i];

                // Transform Feedbackのバッファをバインドする。
                gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, varying.feedbackBuffer!); chk();
            }

            // Transform Feedbackを開始する。
            gl.beginTransformFeedback(gl.POINTS); chk();    // TRIANGLES
        }

        if(pkg.VertexIndexBuffer != undefined){

            // 頂点インデックスバッファをバインドする。
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, pkg.vertexIndexBufferInf!); chk();
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, pkg.VertexIndexBuffer, gl.STATIC_DRAW); chk();

            // 頂点インデックスバッファで描画する。
            gl.drawElements(pkg.mode, pkg.VertexIndexBuffer.length, gl.UNSIGNED_SHORT, 0); chk();
        }
        else{

            if(pkg.numGroup != undefined){

                for(let i = 0; i < pkg.numInput!; i += pkg.numGroup){

                    gl.drawArrays(pkg.mode, i, pkg.numGroup);
                }
            }
            else{

                gl.drawArrays(pkg.mode, 0, pkg.numInput!); chk();
            }
        }

        if (pkg.varyings.length != 0){

            // Transform Feedbackを終了する。
            gl.endTransformFeedback(); chk();

            // すべてのvarying変数に対し
            for (var i = 0; i < pkg.varyings.length; i++) {
                varying = pkg.varyings[i];

                // Transform Feedbackのバッファのバインドを解く。
                gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, null); chk();

                // ARRAY_BUFFERにバインドする。
                gl.bindBuffer(gl.ARRAY_BUFFER, varying.feedbackBuffer!); chk();

                // ARRAY_BUFFERのデータを取り出す。
                gl.getBufferSubData(gl.ARRAY_BUFFER, 0, varying.value as Float32Array); chk();

                // ARRAY_BUFFERのバインドを解く。
                gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
            }

            // Transform Feedbackのバインドを解く。
            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null); chk();

            for(let pipe of pkg.pipes){
                let out_val = pipe.outPackage.args[pipe.outName];
                let in_val  = pipe.inPackage.args[pipe.inName];
                if(out_val instanceof Float32Array){

                    if(in_val instanceof Float32Array){

                        pipe.inPackage.args[pipe.inName] = out_val.slice();
                    }
                    else if(in_val instanceof TextureInfo){

                        assert(in_val.value instanceof Float32Array && in_val.value.length == out_val.length);
                        in_val.value = out_val.slice();
                        in_val.dirty = true;
                    }
                    else{
                        throw new Error();
                    }
                }
                else{
                    throw new Error();
                }
            }
        }

        if(pkg.fragmentShader == GPGPU.minFragmentShader){

            // ラスタライザを有効にする。
            gl.disable(gl.RASTERIZER_DISCARD); chk();
        }

        // プログラムの使用を終了する。
        gl.useProgram(null); chk();

        if(pkg.update != undefined){
            pkg.update();
        }
    }

    /*
        標準のシェーダの文字列をセットします。
    */
    setStandardShaderString() {
    }

    draw(drawable: AbsDrawable, worldMat: Float32Array, viewMat: Float32Array, projMat: Float32Array){

        let modelMat = mat4.create();
        mat4.multiply(modelMat, worldMat, drawable.transform);

        let viewModelMat = mat4.create();
        mat4.multiply(viewModelMat, viewMat, modelMat);

        if(drawable instanceof ComponentDrawable){
            for(let child of drawable.children){
                this.draw(child, modelMat, viewMat, projMat)
            }
            return;
        }

        let pkg = drawable as Package;
        pkg.ready();

        let msec = (new Date()).getTime();
        if(pkg.fps != 0 && msec - pkg.time < 1000 / pkg.fps){
            return;
        }
        pkg.time = msec;

        let projViewModelMat = mat4.create();
        mat4.multiply(projViewModelMat, projMat, viewModelMat);

        let normalMatrix = mat3.create();
        mat3.normalFromMat4(normalMatrix, viewMat);
        // mat4.toInverseMat3(viewMat, normalMatrix);
        // mat3.transpose(normalMatrix);


        pkg.args["uPMVMatrix"] = projViewModelMat;
        pkg.args["uNMatrix"] = normalMatrix;

        if(pkg.args["tick"] == undefined){
            pkg.args["tick"] = 0;
        }
        else{
            pkg.args["tick"]++;
        }

        this.compute(pkg);
    }

    /*
        3D表示をします。
    */
    drawScene() {
        if(this.drawables.length == 0){

            window.requestAnimationFrame(this.drawScene.bind(this));
            return;
        }

        // gl.clearColor(1, 248/255, 220/255, 1); chk();   // cornsilk
        gl.clearColor(0, 0, 0, 1); chk();   // cornsilk
        gl.clearDepth(1.0); chk();                      // Clear everything
        
        gl.enable(gl.DEPTH_TEST); chk();                // Enable depth testing
        gl.depthFunc(gl.LEQUAL);            // Near things obscure far things

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);        

        // カラーバッファと深度バッファをクリアする。
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); chk();

        let viewMat = mat4.create();

        mat4.translate(viewMat, viewMat, [this.drawParam.x, this.drawParam.y, this.drawParam.z]);

        mat4.rotateX(viewMat, viewMat, this.drawParam.xRot);
        mat4.rotateY(viewMat, viewMat, this.drawParam.yRot);
        mat4.rotateZ(viewMat, viewMat, this.drawParam.zRot);

        let projMat = mat4.create();
        mat4.perspective(projMat, 45 * Math.PI / 180, this.canvas.offsetWidth / this.canvas.offsetHeight, 0.1, 1000.0);

        if(this.drawScenelistener != null){
            this.drawScenelistener.beforeDraw();
        }

        for(let drawable of this.drawables){

            let worldMat = mat4.create();
            this.draw(drawable, worldMat, viewMat, projMat);
        }

        if(this.drawScenelistener != null){
            let projViewMat = mat4.create();
            mat4.multiply(projViewMat, projMat, viewMat);

            this.drawScenelistener.afterDraw(projViewMat);
        }

        // 次の再描画でdrawSceneが呼ばれるようにする。
        window.requestAnimationFrame(this.drawScene.bind(this));
    }

    /*
        3D表示を開始します。
    */
    startDraw3D(drawables: AbsDrawable[]) {
        this.drawables = drawables;

        // pointerdownのイベント リスナーを登録する。
        if(this.ui3D.pointerdown != undefined){
            this.canvas.addEventListener("pointerdown", (ev: PointerEvent)=> {
                this.ui3D.pointerdown!(ev, this.drawParam);
            });
        }

        // pointerupのイベント リスナーを登録する。
        if(this.ui3D.pointerup != undefined){
            this.canvas.addEventListener("pointerup", (ev: PointerEvent)=> {
                this.ui3D.pointerup!(ev, this.drawParam);
            });
        }

        // pointermoveのイベント リスナーを登録する。
        if(this.ui3D.pointermove != undefined){
            this.canvas.addEventListener("pointermove", (ev: PointerEvent)=> {
                this.ui3D.pointermove(ev, this.drawParam);
            });
        }

        // mousedownのイベント リスナーを登録する。
        if(this.ui3D.mousedown != undefined){
            this.canvas.addEventListener("mousedown", (ev: MouseEvent)=> {
                this.ui3D.mousedown!(ev, this.drawParam);
            });
        }

        // mouseupのイベント リスナーを登録する。
        if(this.ui3D.mouseup != undefined){
            this.canvas.addEventListener("mouseup", (ev: MouseEvent)=> {
                this.ui3D.mouseup!(ev, this.drawParam);
            });
        }

        // mousemoveのイベント リスナーを登録する。
        if(this.ui3D.mousemove != undefined){
            this.canvas.addEventListener('mousemove', (ev: MouseEvent)=> {
                this.ui3D.mousemove!(ev, this.drawParam);
            });
        }

        // touchmoveのイベント リスナーを登録する。
        if(this.ui3D.touchmove != undefined){
            this.canvas.addEventListener('touchmove', (ev: TouchEvent)=> {
                this.ui3D.touchmove(ev, this.drawParam);
            }, false);
        }

        // wheelのイベント リスナーを登録する。
        if(this.ui3D.wheel != undefined){
            this.canvas.addEventListener("wheel",  (ev: WheelEvent)=> {
                this.ui3D.wheel(ev, this.drawParam);
            });
        }

        // 3D表示をする。
        this.drawScene();
    }
}

/*
    GPGPUのオブジェクトを作ります。

    この関数の内部に関数やクラスを入れて外部から参照されないようにします。
*/
export function CreateGPGPU(canvas: HTMLCanvasElement | undefined = undefined, ui3d: UI3D=new UI3D()) {
    return new GPGPU(canvas, ui3d);
}

}
