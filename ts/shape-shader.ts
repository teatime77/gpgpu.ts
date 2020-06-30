namespace gpgputs {

export class VertexShader {
        static readonly points = `

        // 位置
        in vec3 vertexPosition;

        // 色
        in vec4 vertexColor;

        uniform mat4 uPMVMatrix;
        uniform float pointSize;

        // 色
        out vec4 fragmentColor;

        void main(void) {
            gl_Position   = uPMVMatrix * vec4(vertexPosition, 1.0);
            gl_PointSize  = pointSize;
            fragmentColor = vertexColor;
        }
    `;

    static readonly lines = `

    // 位置
    in vec3 vertexPosition;

    // 色
    in vec4 vertexColor;

    uniform mat4 uPMVMatrix;

    // 色
    out vec4 fragmentColor;

    void main(void) {
        gl_Position   = uPMVMatrix * vec4(vertexPosition, 1.0);
        fragmentColor = vertexColor;
    }
`;

}

export class FragmentShader {
    static readonly points = `
        in  vec4 fragmentColor;
        out vec4 color;

        void main(void) {
            color = fragmentColor;
        }
        `;

}

}
