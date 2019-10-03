namespace gpgpu {

export class Box {
    x1: number;
    x2: number;
    y1: number;
    y2: number;
    z1: number;
    z2: number;

    constructor(x1: number, x2: number, y1: number, y2: number, z1: number, z2: number){
        this.x1 = x1;
        this.x2 = x2;
        this.y1 = y1;
        this.y2 = y2;
        this.z1 = z1;
        this.z2 = z2;
    }

    get width() : number {
        return this.x2 - this.x1;
    }

    get height() : number {
        return this.y2 - this.y1;
    }

    get depth() : number {
        return this.z2 - this.z1;
    }

    // x: number;
    // y: number;
    // z: number;
    // width: number;
    // height: number;
    // depth: number;

    // constructor(x: number, y: number, z: number, width: number, height: number, depth: number){
    // }
}

class Triangle {
    Vertexes: Vertex[];

    constructor(p: Vertex, q: Vertex, r: Vertex, orderd: boolean = false) {
        if (orderd == true) {

            this.Vertexes = [p, q, r];
        }
        else {

            var a = vecSub(q, p);
            var b = vecSub(r, q);

            var c = vecCross(a, b);
            var dir = vecDot(p, c);
            if (0 < dir) {
                this.Vertexes = [p, q, r];
            }
            else {
                this.Vertexes = [q, p, r];
            }
        }
    }
}

class Vertex {
    x: number;
    y: number;
    z: number;
    nx: number;
    ny: number;
    nz: number;
    texX: number;
    texY: number;

    adjacentVertexes: Vertex[];

    constructor(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;

        this.adjacentVertexes = [];
    }
}

class Edge {
    Endpoints: Vertex[];

    constructor(p1: Vertex, p2: Vertex) {
        this.Endpoints = [p1, p2];
    }
}

function sprintf() {
    var args;
    if (arguments.length == 1 && Array.isArray(arguments[0])) {

        args = arguments[0];
    }
    else {

        // 引数のリストをArrayに変換します。
        args = Array.prototype.slice.call(arguments);
    }

    switch (args.length) {
        case 0:
            console.log("");
            return;
        case 1:
            console.log("" + args[1]);
            return;
    }

    var fmt = args[0];
    var argi = 1;

    var output = "";
    var st = 0;
    var k = 0

    for (; k < fmt.length;) {
        var c1 = fmt[k];
        if (c1 = '%' && k + 1 < fmt.length) {
            var c2 = fmt[k + 1];
            if (c2 == 'd' || c2 == 'f' || c2 == 's') {

                output += fmt.substring(st, k) + args[argi];
                k += 2;
                st = k;
                argi++;
                continue;
            }
            else if (c2 == '.' && k + 3 < fmt.length) {

                var c3 = fmt[k + 2];
                var c4 = fmt[k + 3];
                if ("123456789".indexOf(c3) != -1 && c4 == 'f') {
                    var decimal_len = Number(c3);

                    var float_str = '' + args[argi];
                    var period_pos = float_str.indexOf('.');
                    if (period_pos == -1) {
                        float_str += "." + "0".repeat(decimal_len);
                    }
                    else {

                        float_str = (float_str + "0".repeat(decimal_len)).substr(0, period_pos + 1 + decimal_len);
                    }

                    output += fmt.substring(st, k) + float_str;
                    k += 4;
                    st = k;
                    argi++;
                    continue;
                }
            }
        }
        k++;
    }
    output += fmt.substring(st, k);

    return output;
}


function vecLen(p) {
    return Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

function vecDiff(p, q) {
    var dx = p.x - q.x;
    var dy = p.y - q.y;
    var dz = p.z - q.z;

    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function vecSub(a, b) {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function vecDot(a, b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

function vecCross(a, b) {
    return {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x
    };
}

function SetNorm(p: Vertex) {
    var len = vecLen(p);

    if (len == 0) {
        p.nx = 0;
        p.ny = 0;
        p.nz = 0;
    }
    else {

        p.nx = p.x / len;
        p.ny = p.y / len;
        p.nz = p.z / len;
    }
}

function makeRegularIcosahedron() {
    var G = (1 + Math.sqrt(5)) / 2;

    // 頂点のリスト
    var points = [
        new Vertex( 1,  G,  0), // 0
        new Vertex( 1, -G,  0), // 1
        new Vertex(-1,  G,  0), // 2
        new Vertex(-1, -G,  0), // 3

        new Vertex( 0,  1,  G), // 4
        new Vertex( 0,  1, -G), // 5
        new Vertex( 0, -1,  G), // 6
        new Vertex( 0, -1, -G), // 7

        new Vertex( G,  0,  1), // 8
        new Vertex(-G,  0,  1), // 9
        new Vertex( G,  0, -1), // 10
        new Vertex(-G,  0, -1), // 11
    ];

    /*
0 2 4
2 0 5
0 4 8
5 0 10
0 8 10
3 1 6
1 3 7
6 1 8
1 7 10
8 1 10
4 2 9
2 5 11
9 2 11
3 6 9
7 3 11
3 9 11
4 6 8
6 4 9
7 5 10
5 7 11        
    */

    var sphere_r = vecLen(points[0]);

    points.forEach(function (x) {
        console.assert(Math.abs(sphere_r - vecLen(x)) < 0.001);
    });


    // 三角形のリスト
    var triangles = []

    for (var i1 = 0; i1 < points.length; i1++) {
        for (var i2 = i1 + 1; i2 < points.length; i2++) {
            //            println("%.2f : %d %d %.2f", sphere_r, i1, i2, vecDiff(points[i1], points[i2]));

            if (Math.abs(vecDiff(points[i1], points[i2]) - 2) < 0.01) {
                for (var i3 = i2 + 1; i3 < points.length; i3++) {
                    if (Math.abs(vecDiff(points[i2], points[i3]) - 2) < 0.01 && Math.abs(vecDiff(points[i1], points[i3]) - 2) < 0.01) {

                        var pnts = [ points[i1], points[i2], points[i3] ]

                        var tri = new Triangle(pnts[0], pnts[1], pnts[2]);
                        for (var i = 0; i < 3; i++) {
                            pnts[i].adjacentVertexes.push(pnts[(i + 1) % 3], pnts[(i + 2) % 3])
                        }
                            
//                            println("正20面体 %d %d %d", points.indexOf(tri.Vertexes[0]), points.indexOf(tri.Vertexes[1]), points.indexOf(tri.Vertexes[2]))

                        triangles.push(tri);
                    }
                }
            }
        }
    }
    console.assert(triangles.length == 20);

    points.forEach(function (p) {
        // 隣接する頂点の重複を取り除く。
        p.adjacentVertexes = Array.from(new Set(p.adjacentVertexes));

        console.assert(p.adjacentVertexes.length == 5);
    });

    return { points: points, triangles: triangles, sphere_r: sphere_r };
}

function divideTriangle(points, triangles, edges, sphere_r) {
    var divide_cnt = 4;

    for (var divide_idx = 0; divide_idx < divide_cnt; divide_idx++) {

        // 三角形を分割する。
        var new_triangles = [];

        triangles.forEach(function (x) {
            // 三角形の頂点のリスト。
            var pnts = [ x.Vertexes[0], x.Vertexes[1], x.Vertexes[2] ];

            // 中点のリスト
            var midpoints = [];

            for (var i1 = 0; i1 < 3; i1++) {

                // 三角形の2点
                var p1 = pnts[i1];
                var p2 = pnts[(i1 + 1) % 3];

                // 2点をつなぐ辺を探す。
                var edge = edges.find(x => x.Endpoints[0] == p1 && x.Endpoints[1] == p2 || x.Endpoints[1] == p1 && x.Endpoints[0] == p2);
                if (edge == undefined) {
                    // 2点をつなぐ辺がない場合

                    // 2点をつなぐ辺を作る。
                    edge = new Edge(p1, p2);

                    // 辺の中点を作る。
                    edge.Mid = new Vertex((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2);

                    for (var i = 0; i < i; k++) {

                        var k = edge.Endpoints[i].adjacentVertexes.indexOf(edge.Endpoints[(i + 1) % 2]);
                        console.assert(k != -1);
                        edge.Endpoints[i].adjacentVertexes[k] = edge.Mid;
                    }

                    edges.push(edge);
                }

                var mid = edge.Mid;

                midpoints.push(mid);

                var d = vecLen(mid);
                mid.x *= sphere_r / d;
                mid.y *= sphere_r / d;
                mid.z *= sphere_r / d;

                points.push(mid);

                console.assert(Math.abs(sphere_r - vecLen(mid)) < 0.001);
            }

            for (var i = 0; i < 3; i++) {
                var pnt = pnts[i];
                var mid = midpoints[i];

                if (mid.adjacentVertexes.length == 0) {

                    mid.adjacentVertexes.push(pnts[(i + 1) % 3], midpoints[(i + 1) % 3], midpoints[(i + 2) % 3], pnts[i]);
                }
                else {

                    mid.adjacentVertexes.push(pnts[(i + 1) % 3], midpoints[(i + 2) % 3]);
                }
            }

            new_triangles.push(new Triangle(midpoints[0], midpoints[1], midpoints[2], true));
            new_triangles.push(new Triangle(pnts[0], midpoints[0], midpoints[2], true));
            new_triangles.push(new Triangle(pnts[1], midpoints[1], midpoints[0], true));
            new_triangles.push(new Triangle(pnts[2], midpoints[2], midpoints[1], true));
        });

        points.forEach(function (p) {
            console.assert(p.adjacentVertexes.length == 5 || p.adjacentVertexes.length == 6);
        });

        triangles = new_triangles;
    }

    /*

    var new_triangles = [];
    triangles.forEach(function (x) {
        if (x.Vertexes.every(p => p.adjacentVertexes.length == 6)) {
            new_triangles.push(x);
        }
    });
    triangles = new_triangles;
    */


   console.log(`半径:${sphere_r} 三角形 ${triangles.length}`);

    return triangles;
}

function setTextureCoords(points: Vertex[], sphere_r) {
    for (var i = 0; i < points.length; i++) {
        var p = points[i];
        console.assert(i < 12 && p.adjacentVertexes.length == 5 || p.adjacentVertexes.length == 6);

        var x = p.z / sphere_r;
        var y = p.x / sphere_r;
        var z = p.y / sphere_r;

        var th = Math.asin(z);  // [-PI/2 , PI/2]

        p.texY = Math.min(1, Math.max(0, th / Math.PI + 0.5));

        var r = Math.cos(th);

        if (r == 0) {

            p.texX = 0;
            continue;
        }

        x /= r;
        y /= r;

        var ph = Math.atan2(y, x);  // [-PI , PI]

        var u = ph / Math.PI;

        p.texX = Math.min(1, Math.max(0, ph / (2 * Math.PI) + 0.5));
    }
}

export function makeEarthBuffers(tex_inv: TextureInfo) {
    var shape_inf = makeRegularIcosahedron();
    var points = shape_inf.points;
    var triangles = shape_inf.triangles;
    var sphere_r = shape_inf.sphere_r;

    var edges = [];

    triangles = divideTriangle(points, triangles, edges, sphere_r);

    setTextureCoords(points, sphere_r);

    // 頂点インデックス
    var vertexIndices = [];

    triangles.forEach(x =>
        vertexIndices.push(points.indexOf(x.Vertexes[0]), points.indexOf(x.Vertexes[1]), points.indexOf(x.Vertexes[2]))
    );

    // 法線をセット
    points.forEach(p => SetNorm(p));

    // 位置の配列
    var vertices = [];
    points.forEach(p =>
        vertices.push(p.x, p.y, p.z)
    );

    // 法線の配列
    var vertexNormals = [];
    points.forEach(p =>
        vertexNormals.push(p.nx, p.ny, p.nz)
    );

    // テクスチャ座標
    var textureCoords = [];
    points.forEach(p =>
        textureCoords.push(p.texX, p.texY)
    );


    let mesh = {
        vertexPosition: new Float32Array(vertices),
        vertexNormal: new Float32Array(vertexNormals),
        textureCoord: new Float32Array(textureCoords),
        textureImage: tex_inv
    } as Mesh;
    
    let idx_array = new Uint16Array(vertexIndices);

    return [mesh, idx_array];
}


export function makePlaneBuffers(box: Box, nx: number, ny: number, tex_inv: TextureInfo) {
    // 位置の配列
    var vertices = [];

    // 法線の配列
    var vertexNormals = [];

    // テクスチャ座標
    var textureCoords = [];

    const dx = box.width  / (nx - 1);
    const dy = box.height / (ny - 1)
    for (var i = 0; i < ny; i++) {
        var y = i / (ny - 1);
        for (var j = 0; j < nx; j++) {
            var x = j / (nx - 1);

            vertices.push(box.x1 + j * dx, box.y1 + i * dy, 0);
            vertexNormals.push(0, 0, 1);
            textureCoords.push(x, y);
        }
    }

    // 頂点インデックス
    var vertexIndices = [];

    for (var i = 0; i < ny - 1; i++) {
        for (var j = 0; j < nx - 1; j++) {
            var i00 =  i      * nx + j;
            var i01 =  i      * nx + j + 1;
            var i10 = (i + 1) * nx + j;
            var i11 = (i + 1) * nx + j + 1;
            var cnt = vertices.length / 3;
            if(cnt <= i00 || cnt <= i01 || cnt <= i10 || cnt <= i11){
                console.log("");
            }

            vertexIndices.push(i00, i10, i01);
            vertexIndices.push(i01, i10, i11);
        }
    }

    let mesh = {
        vertexPosition: new Float32Array(vertices),
        vertexNormal: new Float32Array(vertexNormals),
        textureCoord: new Float32Array(textureCoords),
        textureImage: tex_inv
    } as Mesh;
    
    let idx_array = new Uint16Array(vertexIndices);

    return [mesh, idx_array];
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

export class Circle extends Drawable {
    constructor(color: Color, numDivision: number){
        super();

        // 位置の配列
        let vertices = [];
    
        // 法線の配列
        let vertexNormals = [];
        
        // 頂点インデックス
        let vertexIndices = [];
        
        // 円周上の点
        for(let idx of range(numDivision)){
            let theta = 2 * Math.PI * idx / numDivision;
            let x = Math.cos(theta);
            let y = Math.sin(theta);

            // 位置
            vertices.push(x, y, 0);

            // 法線
            vertexNormals.push(0, 0, 1);

            // 三角形の頂点インデックス
            vertexIndices.push(idx, (idx + 1) % numDivision, numDivision);
        }

        // 円の中心
        vertices.push(0, 0, 0);
        vertexNormals.push(0, 0, 1);
    
        // 色の配列
        let vertexColors = this.getVertexColors(color, vertices.length);
    
        let mesh = {
            vertexPosition: new Float32Array(vertices),
            vertexNormal: new Float32Array(vertexNormals),
            vertexColor: new Float32Array(vertexColors),
        } as Mesh;
            
        this.param = {
            id: `${this.constructor.name}.${Drawable.count++}`,
            vertexShader: GPGPU.planeVertexShader,
            fragmentShader: GPGPU.planeFragmentShader,
            args: mesh,
            VertexIndexBuffer: new Uint16Array(vertexIndices)
        } as any as PackageParameter;
    }
}

export class Tube extends Drawable {
    constructor(color: Color, numDivision: number){
        super();

        // 位置の配列
        let vertices = [];
    
        // 法線の配列
        let vertexNormals = [];
        
        // 三角形の頂点インデックス
        let vertexIndices = [];
        
        for(let idx of range(numDivision)){
            let theta = 2 * Math.PI * idx / numDivision;
            let x = Math.cos(theta);
            let y = Math.sin(theta);

            vertices.push(x, y, 1);
            vertices.push(x, y, -1);

            vertexNormals.push(x, y, 0);

            let i1 = idx * 2
            let i2 = idx * 2 + 1;
            let i3 = (idx * 2 + 2) % (numDivision * 2);
            let i4 = (idx * 2 + 3) % (numDivision * 2);

            vertexIndices.push(i1, i2, i3);
            vertexIndices.push(i3, i2, i4);
        }
    
        // 色の配列
        let vertexColors = this.getVertexColors(color, vertices.length);

        let mesh = {
            vertexPosition: new Float32Array(vertices),
            vertexNormal: new Float32Array(vertexNormals),
            vertexColor: new Float32Array(vertexColors),
        } as Mesh;
            
        this.param = {
            id: `${this.constructor.name}.${Drawable.count++}`,
            vertexShader: GPGPU.planeVertexShader,
            fragmentShader: GPGPU.planeFragmentShader,
            args: mesh,
            VertexIndexBuffer: new Uint16Array(vertexIndices)
        } as any as PackageParameter;
    }
}

export class Pillar extends ComponentDrawable {
    constructor(colors: Color[], numDivision: number){
        super();

        this.children = [
            new Tube(colors[0], numDivision),
            (new Circle(colors[1], numDivision)).move(0, 0, 1),
            (new Circle(colors[2], numDivision)).move(0, 0, -1),
        ];

        this.param = null;
    }
}

export class Label extends Drawable {
    static texInf: TextureInfo;
    static canvas: HTMLCanvasElement;
    static ctx: CanvasRenderingContext2D;
    static chars: string;
    static readonly cols: number = 32;
    static readonly fontSize: number = 32;

    box : Box;

    static initialize(){
        Label.canvas = document.createElement("canvas");
        Label.canvas.width = Label.cols * Label.fontSize;
        Label.canvas.height = Label.cols * Label.fontSize;

        window.document.body.appendChild(Label.canvas);

        Label.ctx = Label.canvas.getContext('2d');
        Label.ctx.font = `${Label.fontSize}px monospace`;
        Label.ctx.textBaseline = "top";

        Label.chars = "";

        Label.texInf = new TextureInfo(null, null, Label.canvas);
    }

    constructor(text: string, box: Box){
        super();

        if(Label.texInf == undefined){

            Label.initialize();
        }

        this.box = box;

        const textHalfCols = range(text.length).map(i => (text.charCodeAt(i) < 256 ? 1 : 2) as number).reduce((x,y)=>x+y);
        const halfColWidth = box.width / textHalfCols;

        // 位置の配列
        let vertices = [];
    
        // 法線の配列
        let vertexNormals = [];
    
        // テクスチャ座標
        let textureCoords = [];
    
        // 頂点インデックス
        let vertexIndices = [];
    
        let posX = this.box.x1;
        for(let [j, ch] of Array.from(text).entries()){
            let halfFull = (ch.charCodeAt(0) < 256 ? 1 : 2);

            let char_idx = Label.chars.indexOf(ch);
            let new_char = (char_idx == -1)
            if(char_idx == -1){

                char_idx = Label.chars.length;
                Label.chars += ch;
            }

            let iy = Math.floor( char_idx / Label.cols);
            let ix = char_idx % Label.cols;

            if(new_char){

                Label.ctx.fillText(ch, Label.fontSize * ix, Label.fontSize * iy);
            }

            let idx = vertices.length / 3;
            console.assert(vertices.length % 3 == 0);
            for(let i2 = 0; i2 < 2; i2++){
                for(let j2 = 0; j2 < 2; j2++){

                    vertices.push(posX + j2 * halfFull * halfColWidth, this.box.y1 + i2 * this.box.height, 0);
                    vertexNormals.push(0, 0, 1);

                    let x = (ix + j2 * halfFull / 2) / Label.cols;
                    let y = (Label.cols - 1 - iy + i2) / Label.cols;
                    textureCoords.push(x , y);
                }
            }

            let i00 = idx;
            let i01 = idx + 1;
            let i10 = idx + 2;
            let i11 = idx + 3;
            let cnt = vertices.length / 3;
            if(cnt <= i00 || cnt <= i01 || cnt <= i10 || cnt <= i11){
                console.log("");
            }

            vertexIndices.push(i00, i10, i01);
            vertexIndices.push(i01, i10, i11);

            posX += halfFull * halfColWidth;
        }
    
        let mesh = {
            vertexPosition: new Float32Array(vertices),
            vertexNormal: new Float32Array(vertexNormals),
            textureCoord: new Float32Array(textureCoords),
            textureImage: new gpgpu.TextureInfo(null, null, Label.canvas)
        } as Mesh;
        
    
        this.param = {
            id: `label${Drawable.count++}`,
            vertexShader: gpgpu.GPGPU.planeTextureVertexShader,//.textureSphereVertexShader,
            fragmentShader: gpgpu.GPGPU.planeTextureFragmentShader,//defaultFragmentShader,
            args: mesh,
            VertexIndexBuffer: new Uint16Array(vertexIndices)
        } as any as PackageParameter;

    }
}


export class ImageDrawable extends Drawable {
    img: HTMLImageElement;

    constructor(src: string, fnc) {
        super();
        this.img = new Image();
        this.img.onload = fnc;
        this.img.src = src;
    }

    getParam() {
        if (!this.param) {

            var [mesh, idx_array] = gpgpu.makePlaneBuffers(new gpgpu.Box(-1, -0.5, -1, -0.5, 0, 0), 11, 11, new gpgpu.TextureInfo(null, null, this.img));

            this.param = {
                id: "Earth",
                vertexShader: gpgpu.GPGPU.textureSphereVertexShader,
                fragmentShader: gpgpu.GPGPU.defaultFragmentShader,
                args: mesh,
                VertexIndexBuffer: idx_array
            } as any as PackageParameter;
        }

        return this.param;
    }
}


}