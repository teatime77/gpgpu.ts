import {Drawable, range, GPGPU, TextureInfo, Package, CreateGPGPU, Color, Points, Lines, Vertex} from "./gpgpu.js";
import { makePlaneBuffers, ImageDrawable, Circle, Tube, Pillar, Cone, RegularIcosahedron, GeodesicPolyhedron, Label, Box } from "./shape.js";

let mygpgpu;

class TextDrawable extends Drawable {
    canvas_2d;
    box;

    constructor(box){
        super();
        this.box = box;
        this.canvas_2d = document.getElementById("canvas-2d");
        let ctx = this.canvas_2d.getContext('2d');
        ctx.font = "16px monospace";
        ctx.textBaseline = "top";

        let text = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()=+-*/~|@;:,.[]{}";
        text += "あいうえおかきくけこさしすせそたちつてとなにぬねの";

        for(let y = 0; y < 16; y++){
            let base = 16 * y;
            if(text.length <= base){
                break;
            }
            let s = text.substring(base, base + 16);
            s = range(s.length).map(i => s.charCodeAt(i) < 256 ? s[i] + " " : s[i]).join("");
            ctx.fillText(s, 0, 16 * y);
        }
    }

    onDraw() {
        if (!this.package) {

            var [mesh, idx_array] = makePlaneBuffers(this.box, 11, 11, new TextureInfo(null, null, this.canvas_2d));

            this.package = {
                id: "label",
                vertexShader: GPGPU.textureSphereVertexShader,
                fragmentShader: GPGPU.defaultFragmentShader,
                args: mesh,
                VertexIndexBuffer: idx_array
            } as Package ;
        }

        return this.package;
    }
}

let imageDrawable;
let textDrawable;
let textDrawable2;
let labelDrawable1;
let labelDrawable2;

export function testBodyOnLoad(){
    imageDrawable = new ImageDrawable("../img/world.topo.bathy.200408.2048x2048.png", ()=>{
        var canvas = document.getElementById("webgl-canvas") as HTMLCanvasElement;
        mygpgpu = CreateGPGPU(canvas);
        mygpgpu.startDraw3D([ 
            imageDrawable,
            labelDrawable1,
            labelDrawable2,
            (new Circle(new Color(1,0,0,1), 20)).scale(0.2, 0.1, 0.2).move(1, 0, 0.5),
            (new Tube(new Color(0,1,0,1), 20)).scale(0.1, 0.1, 2).move(-1, 0, 0),
            (new Pillar([Color.red, Color.green, Color.blue], 20)).scale(0.1, 0.1, 1).move(0, 3, 0),
            (new Cone(Color.red, 20)).scale(0.2, 0.2, 1).move(2, 0, 0.5),
            new Points(new Float32Array([1.5, -1.3, 0, -1.5, -1.3, 0]), new Float32Array([1,0,0,1, 0,0,1,1]), 5),
            new Lines([{x:1.5,y:-1.5,z:0} as Vertex,{x:-1.5,y:-1.5,z:0} as Vertex], Color.blue),
            (new RegularIcosahedron(new Color(0,1,0,1))).scale(0.3, 0.3, 0.3).move(2, -2, 0),
            (new GeodesicPolyhedron(new Color(0,0,1,1), 1)).scale(0.3, 0.3, 0.3).move(3,  2, 0),
            (new GeodesicPolyhedron(new Color(0,0,1,1), 2)).scale(0.3, 0.3, 0.3).move(1.5,  1, 0),
            (new GeodesicPolyhedron(new Color(0,0,1,1), 3)).scale(0.3, 0.3, 0.3).move(-1.5, -1, 0),
            (new GeodesicPolyhedron(new Color(0,0,1,1), 4)).scale(0.3, 0.3, 0.3).move(-3, -2, 0),
        ]);
        // mygpgpu.startDraw3D([ imageDrawable, textDrawable, textDrawable2 ]);
    });

    textDrawable   = new TextDrawable(new Box(0.5, 1, 0.5, 1, 0, 0));
    textDrawable2  = new TextDrawable(new Box(-0.5, 0, 0.5, 1, 0, 0));

    let s = "0123456789";

    for(let i = "A".charCodeAt(0); i <= "z".charCodeAt(0); i++){
        s += String.fromCharCode(i);
    }

    let n = "ぁ".charCodeAt(0);
    for(let i = 0; i < 256; i++){
        s += String.fromCharCode(n + i);
        if(s.length == 256){
            break;
        }
    }

    new Label(s, new Box(-0.2, 0.5, -0.2, 0.5, 0, 0))
    labelDrawable1 = new Label("012ポマミこんにちは", new Box(-2, 2, 1, 2, 0, 0))
    labelDrawable2 = new Label("012WX Y漢字α常用漢字", new Box(-2, 2, 0, 0.9, 0, 0))
}


export function testImageDrawable(){
    imageDrawable = new ImageDrawable("img/world.topo.bathy.200408.2048x2048.png", ()=>{
        var canvas = document.getElementById("webgl-canvas") as HTMLCanvasElement;
        mygpgpu = CreateGPGPU(canvas);
        mygpgpu.startDraw3D([ 
            imageDrawable,
        ]);
    });
}
