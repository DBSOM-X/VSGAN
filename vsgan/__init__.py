#####################################################################
#                         Created by PRAGMA                         #
#                 https://github.com/imPRAGMA/VSGAN                 #
#####################################################################
#              For more details, consult the README.md              #
#####################################################################

import functools

import mvsfunc
import numpy as np
import torch
#import pkgutil
import vapoursynth as vs
from vapoursynth import core


class VSGAN:

    def __init__(self, device="cuda"):
        #Check for a Google TPU. Not needed now, as I can't get the any ESRGANmodels to load with it...  
        #if pkgutil.find_loader("torch_xla"):
        #    import torch_xla.core.xla_model as xm
        #    self.torch_device = xm.xla_device()
        self.torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Stubs
        self.model_file = None
        self.model_scale = None
        self.rrdb_net_model = None

    def load_model(self, model, scale):
        self.model_file = model
        self.model_scale = scale
        # attempt to use New Arch, and if that fails, attempt to use Old Arch
        # if both fail to be loaded, it will raise it's original exception
        for arch in range(2):
            self.rrdb_net_model = self.get_rrdb_net_arch(arch)
            try:
                self.rrdb_net_model.load_state_dict(torch.load(self.model_file), strict=True)
                break
            except RuntimeError:
                if arch == 1:
                    raise
        self.rrdb_net_model.eval()
        self.rrdb_net_model = self.rrdb_net_model.to(self.torch_device)

    def run(self, clip, chunk=False, reconvert=False, pad = 16):
        
        #Makes a horizontal or vertical grayscale, FP32 gradient ramp VS clip
        #White is on the right/bottom, black is on the left/top
        #Based on https://forum.doom9.org/showthread.php?p=1885013
        def make_gradient(WIDTH, HEIGHT, length, vertical = False):
            #https://stackoverflow.com/questions/4337902/how-to-fill-opencv-image-with-one-solid-color
            def create_blank(width, height, color=(0, 0, 0)):
                image = np.zeros((height, width, 3), np.uint8)
                #color = tuple(reversed(color))
                image[:] = color
                return image
                
            #https://github.com/KotoriCANOE/MyTF/blob/master/utils/vshelper.py
            def float32_vsclip(s, clip=None):
                assert isinstance(s, np.ndarray)
                core = vs.get_core()
                if len(s.shape) <= 3: s = s.reshape([1] + list(s.shape))
                num = s.shape[-4]
                height = s.shape[-3]
                width = s.shape[-2]
                planes = 1
                if clip is None:
                    clip = core.std.BlankClip(None, width, height, vs.GRAYS, num)
            
                def convert_func(n, f):
                    fout = f.copy()
                    for p in range(planes):
                        d = np.array(fout.get_write_array(p), copy=False)
                        np.copyto(d, s[n, :, :, p])
                        del d
                    return fout
                return core.std.ModifyFrame(clip, clip, convert_func)
            
            color1 = (0)
            color2 = (255)
            img1 = create_blank(WIDTH, HEIGHT, color1).astype(np.float32)/255.0
            img2 = create_blank(WIDTH, HEIGHT, color2).astype(np.float32)/255.0
            if vertical:
                c = np.linspace(0, 1, HEIGHT)[:, None, None]
            else:
                c = np.linspace(0, 1, WIDTH)[None,:, None] 
            gradient = img1 + (img2 - img1) * c
            clip = float32_vsclip(gradient)
            return core.std.Loop(clip, length)
        
        #Like stackhorizontal, but with padding
        def merge_horizontal(left, right):
            lwidth = left.width - pad
            rwidth = right.width - pad
            if (left.height, left.num_frames, left.format) != (right.height, right.num_frames, right.format):
                raise Exception("Left and right clip must have the same height, format and number of frames!")
            mask = make_gradient(pad * 2 * self.model_scale, left.height, left.num_frames)
            mask = core.std.AddBorders(mask, left = lwidth - pad * self.model_scale, color = [0.0])
            mask = core.std.AddBorders(mask, right = rwidth - pad * self.model_scale, color = [1.0])

            #raise Exception(lwidth, rwidth)
            left = core.std.AddBorders(left, right = rwidth - pad * self.model_scale)
            right = core.std.AddBorders(right, left = lwidth - pad * self.model_scale)
            #raise Exception(left.width, left.height, right.width, right.height, mask.width, mask.height, mask.format)
            return core.std.MaskedMerge(left, right, mask)

        #Like stackvertical, but with padding
        def merge_vertical(top, bottom):
            theight = top.height - pad
            bheight = bottom.height - pad
            if (top.width, top.num_frames, top.format) != (bottom.width, bottom.num_frames, bottom.format):
                raise Exception("Top and bottom clip must have the same height, format and number of frames!")
            mask = make_gradient(top.width, pad * 2 * self.model_scale, top.num_frames, vertical = True)
            mask = core.std.AddBorders(mask, top = theight - pad * self.model_scale, color = [0.0])
            mask = core.std.AddBorders(mask, bottom = bheight - pad * self.model_scale, color = [1.0])

            top = core.std.AddBorders(top, bottom = bheight - pad * self.model_scale)
            bottom = core.std.AddBorders(bottom, top = theight - pad * self.model_scale)
            return core.std.MaskedMerge(top, bottom, mask)

        # remember the clip's original format
        original_format = clip.format
        # convert clip to RGB24 as it cannot read any other color space
        buffer = mvsfunc.ToRGB(clip, depth=8)  # expecting RGB24 8bit
        # send the clip array to execute()
        results = []
        for c in self.chunk_clip(buffer, pad) if chunk else [buffer]:
            results.append(core.std.FrameEval(
                core.std.BlankClip(
                    clip=c,
                    width=c.width * self.model_scale,
                    height=c.height * self.model_scale
                ),
                functools.partial(
                    self.execute,
                    clip=c
                )
            ))
        # if chunked, rejoin the chunked clips, otherwise return the result
        if chunk:
            tophalf = merge_horizontal(core.resize.Spline36(results[0], format = vs.RGBS), core.resize.Spline36(results[1], format = vs.RGBS))
            bottomhalf = merge_horizontal(core.resize.Spline36(results[2], format = vs.RGBS), core.resize.Spline36(results[3], format = vs.RGBS))
            buffer = merge_vertical(tophalf, bottomhalf)
        else:
            buffer = core.resize.Spline36(results[0], format = vs.RGBS)
        # Optionally convert back to the original color space
        # Note that (unlike ToRGB), mvs.ToYUV "guesses" the original color space
        # Based on whether the RGB clip is SD, HD, or UHD
        # Upscaling is going to influence this decision, 
        # so it's usually best to leave the output unchanged
        # And let the user reconvert it with the appropriate arguments. 
        if reconvert and (original_format.color_family != buffer.format.color_family):
            if original_format.color_family == vs.ColorFamily.RGB:
                buffer = mvsfunc.ToRGB(buffer)
            if original_format.color_family == vs.ColorFamily.YUV:
                buffer = mvsfunc.ToYUV(buffer, css=original_format.name[3:6])
        # return the new frame
        return buffer

    def get_rrdb_net_arch(self, arch):
        """
        Import Old or Current Era RRDB Net Architecture
        """
        if arch == 0:
            from . import RRDBNet_arch_old as Arch
            return Arch.RRDB_Net(
                3, 3, 64, 23,
                gc=32,
                upscale=self.model_scale,
                norm_type=None,
                act_type="leakyrelu",
                mode="CNA",
                res_scale=1,
                upsample_mode="upconv"
            )
        from . import RRDBNet_arch as Arch
        return Arch.RRDBNet(3, 3, 64, 23, gc=32)

    @staticmethod
    def chunk_clip(clip, pad):
        #Split the image into 4 images, with some overlap over the seams
        hcrop = clip.width/2 - pad
        vcrop = clip.height/2 - pad
        return [
            core.std.Crop(clip, bottom = vcrop, right = hcrop),
            core.std.Crop(clip, bottom = vcrop, left = hcrop),
            core.std.Crop(clip, top = vcrop, right = hcrop),
            core.std.Crop(clip, top = vcrop, left = hcrop)
        ]

    @staticmethod
    def cv2_imread(frame, plane_count):
        """
        Alternative to cv2.imread() that will directly read images to a numpy array
        """
        return np.dstack(
            [np.array(frame.get_read_array(i), copy=False) for i in reversed(range(plane_count))]
        )

    @staticmethod
    def cv2_imwrite(image, out_color_space="RGB24"): #TODO: "RGBS"
        """
        Alternative to cv2.imwrite() that will convert the data into an image readable by VapourSynth
        """
        if len(image.shape) <= 3:
            image = image.reshape([1] + list(image.shape))
        # Define the shapes items
        plane_count = image.shape[-1]
        image_width = image.shape[-2]
        image_height = image.shape[-3]
        image_length = image.shape[-4]
        # this is a clip (or array buffer for frames) that we will insert the GAN'd frames into
        buffer = core.std.BlankClip(
            clip=None,
            width=image_width,
            height=image_height,
            format=vs.PresetFormat[out_color_space],
            length=image_length
        )

        def replace_planes(n, f):
            frame = f.copy()
            for i, plane_num in enumerate(reversed(range(plane_count))):
                # todo ; any better way to do this without storing the np.array in a variable?
                # todo ; perhaps some way to directly copy it to s?
                d = np.array(frame.get_write_array(plane_num), copy=False)
                # copy the value of d, into s[frame_num, :, :, plane_num]
                np.copyto(d, image[n, :, :, i], casting="unsafe")
                # delete the d variable from memory
                del d
            return frame

        # take the blank clip and insert the new data into the planes and return it back to sender
        return core.std.ModifyFrame(clip=buffer, clips=buffer, selector=replace_planes)

    def execute(self, n, clip):
        """
        Essentially the same as ESRGAN, except it replaces the cv2 functions with ones geared towards VapourSynth
        https://github.com/xinntao/ESRGAN/blob/master/test.py#L26
        """
        # get the frame being used
        frame = clip.get_frame(n)
        img = self.cv2_imread(frame=frame, plane_count=clip.format.num_planes)
        img = img * 1.0 / 255 #TODO: Remove
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_lr = img.unsqueeze(0)
        img_lr = img_lr.to(self.torch_device)
        with torch.no_grad():
            output = self.rrdb_net_model(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round() #TODO: Remove
        return self.cv2_imwrite(image=output, out_color_space=clip.format.name)
