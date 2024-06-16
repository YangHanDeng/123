import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstAudio', '1.0')

from gi.repository import Gst, GObject, GstBase, GstAudio, GLib

import numpy as np
from modelscope.pipelines import pipeline

DEFAULT_CHANNEL=2
FLOAT_BYTE_RATIO=4
DEFAULT_WINDOW_SIZE=32
FIXED_CAPS = Gst.Caps.from_string('audio/x-raw, format=F32LE, layout=interleaved, rate=16000, channels={ch}'.format(ch=DEFAULT_CHANNEL))

DO_MIX = False
DO_NORMALIZE = False
MIX_RATIO = 0.005
DO_COMPRESS = False

class FRCRN(GstBase.BaseTransform):
    __gstmetadata__ = ('FRCRN Python','Transform',
                      'Test Combination of FRCRN and Gstreamer', 'IONetwork Henry')
    

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                           Gst.PadDirection.SRC,
                                           Gst.PadPresence.ALWAYS,
                                           FIXED_CAPS),
                       Gst.PadTemplate.new("sink",
                                           Gst.PadDirection.SINK,
                                           Gst.PadPresence.ALWAYS,
                                           FIXED_CAPS))
    __gproperties__ = {
        "window-size": (int,
                 "Window Size",
                 "Number of enhance model's input chunk",
                 2,
                 GLib.MAXINT,
                 DEFAULT_WINDOW_SIZE,
                 GObject.ParamFlags.READWRITE
                )
    }
    def __init__(self):
        self.info = GstAudio.AudioInfo()
        self.model = pipeline(
            'acoustic-noise-suppression',
            model='./speech_frcrn_ans_cirm_16k_official_result')
        self.window_size = DEFAULT_WINDOW_SIZE
        self.i=0
        self.window_func = np.array([])
        self.drybuf=np.empty([[]]*DEFAULT_CHANNEL, dtype=np.float32)
        self.wetbuf=np.empty([[]]*DEFAULT_CHANNEL, dtype=np.float32)

    def do_get_property(self, prop):
        if prop.name == 'window-size':
            return self.window_size
        
    def do_set_property(self, prop, value):
        if prop.name == 'window-size':
            self.window_size = value

    def _crossfade(self, window_func, overlap_size, sound1, sound2):
        mix_sound = np.add(
            np.multiply(sound1, window_func[:,-int(overlap_size) :]),
            np.multiply(sound2, window_func[:,0:overlap_size])
        )
        return mix_sound
    def _normalize(self):
        std = self.drybuf.std(dim=1).reshape((2,1))
        self.drybuf = (
            np.add (self.drybuf, -self.drybuf.mean(dim=1).reshape((2,1))) / (1e-3 + std)
        )
    

    def do_transform_ip(self, buf): # hanning with 64 chunks, start with all empty array
        try:
            with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:
                # info.data's type is memoryview, info.size is in byte size, 
                arrlen = info.size//DEFAULT_CHANNEL//FLOAT_BYTE_RATIO
                self.window_func = np.array([np.hanning(arrlen*2)]*DEFAULT_CHANNEL)
                if self.i == 0:
                    self.drybuf = np.zeros((DEFAULT_CHANNEL, self.window_size * arrlen))
                    self.wetbuf = np.zeros((DEFAULT_CHANNEL, self.window_size * arrlen))
                
                # interleaved => order='F'
                numpy_data = np.ndarray(shape = (DEFAULT_CHANNEL, arrlen),
                                        dtype = np.float32, buffer = info.data, order='F')
                self.drybuf = np.hstack((self.drybuf, numpy_data))
                if DO_NORMALIZE:
                    self._normalize()

                self.drybuf = self.drybuf[:, arrlen:]
                enh_bytes = self.model(self.drybuf)['output_pcm']
                enhanced = np.ndarray(shape = (DEFAULT_CHANNEL, arrlen),
                                        dtype = np.float32, buffer = enh_bytes, order='F') # enhance(self.model, self.df_state, self.drybuf)

                #print("enhanced{idx}: {val}".format(idx=self.i, val=enhanced[:,-arrlen:]))
                # if DO_MIX:
                #     enhanced = MIX_RATIO*self.drybuf + (1-MIX_RATIO)*enhanced
                
                # if DO_COMPRESS:
                #     enhanced = 0.99 * np.tanh(enhanced)
                
                self.wetbuf = np.hstack((
                    self.wetbuf[:,arrlen:-arrlen],
                    self._crossfade(self.window_func, arrlen, self.wetbuf[:,-arrlen:], enhanced[:,-2*arrlen:-arrlen]),
                    enhanced[:,-arrlen:]
                ))
                numpy_data[:]=self.wetbuf[:,-2*arrlen:-arrlen] # copy result to output
                #save_audio("testdir/enhanced{idx}.wav".format(idx=self.i), self.wetbuf[:,-2*arrlen:-arrlen], self.df_state.sr())
                self.i +=1
                return Gst.FlowReturn.OK
        except Gst.MapError as e:
            Gst.error("Mapping error: %s" % e)
            return Gst.FlowReturn.ERROR


'''
if __name__ == "__main__":
    # Load default model
    model, df_state, _ = init_df(model_base_dir="DeepFilterNet2") # copied
    # Download and open some audio file. You use your audio files here
    
    audio, _ = load_audio("noise1.wav", sr=df_state.sr())
    # Denoise the audio
    enhanced = enhance(model, df_state, audio)
    # Save for listening
    save_audio("enhanced.wav", enhanced, df_state.sr())
'''

GObject.type_register(FRCRN)
__gstelementfactory__ = ("FRCRN", Gst.Rank.NONE, FRCRN)
