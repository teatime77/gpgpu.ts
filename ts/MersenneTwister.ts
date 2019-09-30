/*
 * NOTE :
 * 	This program is a JavaScript version of Mersenne Twister,
 * 	conversion from the original program (mt19937ar.c),
 * 	translated by yunos on december, 6, 2008.
 * 	If you have any questions about this program, please ask me by e-mail.
 * 
 * 
 * 
 * Updated 2008/12/08
 * Ver. 1.00
 * charset = UTF8
 * 
 * Mail : info@graviness.com
 * Home : http://www.graviness.com/
 * 
 * �[�����������탁���Z���k�E�c�C�X�^�N���X�D
 * 
 * Math�N���X�̃N���X���\�b�h��mersenneTwisterRandom���\�b�h��ǉ����܂��D
 * 
 * Ref.
 * 	http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/mt.html
 */



/**
 * �[�����������탁���Z���k�E�c�C�X�^�N���X�D
 * 
 * �[�������������@�̕W���ł��郁���Z���k�E�c�C�X�^����������܂��D
 * 
 * ��������32�r�b�g�����^�̈�l��������{�Ƃ��C��������46�r�b�g�����^��l�����C
 * ���������_�^�̈�l�����𐶐����܂��D
 * ���������̏������ɂ́C��̐������g�p���܂����C�K�v�ɉ�����
 * �z���p�����C�Ӄr�b�g���̒l���g�p���邱�Ƃ��ł��܂��D
 * 
 * ���̃N���X�͈ȉ��̃T�C�g(C����\�[�X)��JavaScript����ڐA�łł��D
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c
 * (http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/mt.html)
 * �O���C���^�t�F�[�X�́CJava��java.util.Random�N���X���Q�l�Ɏ�������Ă��܂��D
 * http://sdc.sun.co.jp/java/docs/j2se/1.4/ja/docs/ja/api/java/util/Random.html
 * 
 * ���\�́C�r���g�C����Math.random�̖�2���̈�ł����C
 * �����̕i���͓��Y�T�C�g�Ɏ����ʂ�ł��D
 * 
 * �g�p��)
 * // �C���X�^���X�𐶐����C��������������ݎ����ŏ��������܂��D
 * var mt = new MersenneTwister(new Date().getTime());
 * for (var i = 0; i < 1000; ++i) {
 * 	// 32�r�b�g�������������^�̈�l����
 * 	var randomNumber = mt.nextInteger();
 * }
 */
function class__MersenneTwister__(window) {
    var className = "MersenneTwister";

    var $next = "$__next__";

    var N = 624;
    var M = 397;
    var MAG01 = [0x0, 0x9908b0df];

    /**
	 * �V���������W�F�l���[�^�𐶐����܂��D
	 * �����ɉ������V�[�h��ݒ肵�܂��D
	 * 
	 * @param (None)	�V���������W�F�l���[�^�𐶐����܂��D
	 * �V�[�h�͌��ݎ������g�p���܂��D
	 * @see Date#getTime()
	 * ---
	 * @param number	
	 * @see #setSeed(number)
	 * ---
	 * @param number[]	
	 * @see #setSeed(number[])
	 * ---
	 * @param number, number, ...	
	 * @see #setSeed(number, number, ...)
	 */
    var F = window[className] = function () {
        this.mt = new Array(N);
        this.mti = N + 1;

        var a = arguments;
        switch (a.length) {
            case 0:
                this.setSeed(new Date().getTime());
                break;
            case 1:
                this.setSeed(a[0]);
                break;
            default:
                var seeds = new Array();
                for (var i = 0; i < a.length; ++i) {
                    seeds.push(a[i]);
                }
                this.setSeed(seeds);
                break;
        }
    };

    var FP = F.prototype;

    /**
	 * �����W�F�l���[�^�̃V�[�h��ݒ肵�܂��D
	 * 
	 * @param number	�P��̐��l���g�p���C
	 * 	�����W�F�l���[�^�̃V�[�h��ݒ肵�܂��D
	 * ---
	 * @param number[]	�����̐��l���g�p���C
	 * 	�����W�F�l���[�^�̃V�[�h��ݒ肵�܂��D
	 * ---
	 * @param number, number, ...	�����̐��l���g�p���C
	 * 	�����W�F�l���[�^�̃V�[�h��ݒ肵�܂��D
	 */
    FP.setSeed = function () {
        var a = arguments;
        switch (a.length) {
            case 1:
                if (a[0].constructor === Number) {
                    this.mt[0] = a[0];
                    for (var i = 1; i < N; ++i) {
                        var s = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
                        this.mt[i] = ((1812433253 * ((s & 0xffff0000) >>> 16))
                                << 16)
                            + 1812433253 * (s & 0x0000ffff)
                            + i;
                    }
                    this.mti = N;
                    return;
                }

                this.setSeed(19650218);

                var l = a[0].length;
                var i = 1;
                var j = 0;

                for (var k = N > l ? N : l; k != 0; --k) {
                    var s = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30)
                    this.mt[i] = (this.mt[i]
                            ^ (((1664525 * ((s & 0xffff0000) >>> 16)) << 16)
                                + 1664525 * (s & 0x0000ffff)))
                        + a[0][j]
                        + j;
                    if (++i >= N) {
                        this.mt[0] = this.mt[N - 1];
                        i = 1;
                    }
                    if (++j >= l) {
                        j = 0;
                    }
                }

                for (var k = N - 1; k != 0; --k) {
                    var s = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
                    this.mt[i] = (this.mt[i]
                            ^ (((1566083941 * ((s & 0xffff0000) >>> 16)) << 16)
                                + 1566083941 * (s & 0x0000ffff)))
                        - i;
                    if (++i >= N) {
                        this.mt[0] = this.mt[N - 1];
                        i = 1;
                    }
                }

                this.mt[0] = 0x80000000;
                return;
            default:
                var seeds = new Array();
                for (var i = 0; i < a.length; ++i) {
                    seeds.push(a[i]);
                }
                this.setSeed(seeds);
                return;
        }
    };

    /**
	 * ���̋[�������𐶐����܂��D
	 * @param bits	�o�͒l�̗L���r�b�g�����w�肵�܂��D
	 * 	0 &lt; bits &lt;= 32�Ŏw�肵�܂��D
	 * @param ���̋[�������D
	 */
    FP[$next] = function (bits) {
        if (this.mti >= N) {
            var x = 0;

            for (var k = 0; k < N - M; ++k) {
                x = (this.mt[k] & 0x80000000) | (this.mt[k + 1] & 0x7fffffff);
                this.mt[k] = this.mt[k + M] ^ (x >>> 1) ^ MAG01[x & 0x1];
            }
            for (var k = N - M; k < N - 1; ++k) {
                x = (this.mt[k] & 0x80000000) | (this.mt[k + 1] & 0x7fffffff);
                this.mt[k] = this.mt[k + (M - N)] ^ (x >>> 1) ^ MAG01[x & 0x1];
            }
            x = (this.mt[N - 1] & 0x80000000) | (this.mt[0] & 0x7fffffff);
            this.mt[N - 1] = this.mt[M - 1] ^ (x >>> 1) ^ MAG01[x & 0x1];

            this.mti = 0;
        }

        var y = this.mt[this.mti++];
        y ^= y >>> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >>> 18;
        return y >>> (32 - bits);
    };

    /**
	 * ��l���z��boolean�^�̋[��������Ԃ��܂��D
	 * @return true or false�D
	 */
    FP.nextBoolean = function () {
        return this[$next](1) == 1;
    };

    /**
	 * ��l���z�̕�����32�r�b�g�����^�̋[��������Ԃ��܂��D
	 * @return ������32�r�b�g�����^�̋[�������ŁC0�ȏ�4294967295�ȉ��ł��D
	 */
    FP.nextInteger = function () {
        return this[$next](32);
    };

    /**
	 * ��l���z�̕�����46�r�b�g�����^�̋[��������Ԃ��܂��D
	 * @return ������46�r�b�g�����^�̋[�������ŁC0�ȏ�70368744177663�ȉ��ł��D
	 */
    FP.nextLong = function () {
        // NOTE: 48�r�b�g�ȏ�Ōv�Z���ʂ��������D
        // (46 - 32) = 14 = [7] + [7], 32 - [7] = [25], 32 - [7] = [25]
        // 2^(46 - [25]) = 2^21 = [2097152]
        return this[$next](25) * 2097152 + this[$next](25);
    };

    /**
	 * 0.0�`1.0�͈̔͂ň�l���z��32�r�b�g�x�[�X��
	 * ���������_�^�̋[��������Ԃ��܂��D
	 * @return ���J��Ԃ�[0.0 1.0)�ł��D
	 */
    FP.nextFloat = function () {
        return this[$next](32) / 4294967296.0; // 2^32
    };

    /**
	 * 0.0�`1.0�͈̔͂ň�l���z��46�r�b�g�x�[�X��
	 * ���������_�^�̋[��������Ԃ��܂��D
	 * @return ���J��Ԃ�[0.0 1.0)�ł��D
	 */
    FP.nextDouble = function () {
        return (this[$next](25) * 2097152 + this[$next](25))
			/ 70368744177664.0; // 2^46
    };

} class__MersenneTwister__(window);



/**
 * �[�����������Ƀ����Z���k�E�c�C�X�^���g�p���C���J���[0 1.0)��
 * ���������_�^�̋[�������𐶐����܂��D
 * Math.random�Ɠ��l�Ɏg�p���܂��D
 * 
 * �g�p��)
 * // 0�ȏ�1��菬�����s�������_�^�̒l�𐶐����܂��D
 * var r = Math.mersenneTwisterRandom();
 */
Math.mersenneTwisterRandom = function () {
    Math.__MERSENNE_TWISTER__ = new MersenneTwister();

    return function () {
        return Math.__MERSENNE_TWISTER__.nextFloat();
    }
}();

var theMersenneTwister = new MersenneTwister(0);
var MersenneTwisterIdx = 0;
function Math_random() {
    MersenneTwisterIdx++;
    return theMersenneTwister.nextFloat();
}