#####################################################################
#                         Created by PRAGMA                         #
#                 https://github.com/imPRAGMA/VSGAN                 #
#####################################################################
#              For more details, consult the README.md              #
#####################################################################

import functools
import numpy as np
import torch
import vapoursynth as vs
from vapoursynth import core


class VSGAN:

    def __init__(self, device="cuda"):
        #Check for a Google TPU. Not needed now, as ESRGANmodels seemingly won't load with it...  
        import pkgutil
        if pkgutil.find_loader("torch_xla"):
            import torch_xla.core.xla_model as xm
            self.torch_device = xm.xla_device()
        else:
            self.torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Stubs
        self.model_file = None
        self.model_scale = None
        self.rrdb_net_model = None

    def load_model(self, model):
        state_dict = torch.load(model)

        # Check if new-arch and convert
        if 'conv_first.weight' in state_dict:
            state_dict = self.convert_new_to_old(state_dict)

        # extract model information
        scale2 = 0
        max_part = 0
        scalemin = 6
        for part in list(state_dict):
            parts = part.split('.')
            n_parts = len(parts)
            if n_parts == 5 and parts[2] == 'sub':
                nb = int(parts[3])
            elif n_parts == 3:
                part_num = int(parts[1])
                if part_num > scalemin and parts[0] == 'model' and parts[2] == 'weight':
                    scale2 += 1
                if part_num > max_part:
                    max_part = part_num
                    out_nc = state_dict[part].shape[0]
        upscale = 2 ** scale2
        in_nc = state_dict['model.0.weight'].shape[1]
        nf = state_dict['model.0.weight'].shape[0]
        self.model_scale = upscale

        self.rrdb_net_model = self.get_rrdb_net_arch(in_nc, out_nc, nf, nb)
        self.rrdb_net_model.load_state_dict(state_dict, strict=False)
        self.rrdb_net_model.eval()

    #"Reconvert" specifies whether the output should be converted to the input format, if it's not already single precision RGB
    #"Pad" is the number of pixels around each pixel that will overlap in chunk mode, to prevent seams. 
    #"Matrix_s" is the target colorspace for RGB -> YUV. See http://www.vapoursynth.com/doc/functions/resize.html
    def run(self, clip, chunk=False, pad = 16, reconvert=False, matrix_s = "709"):
        
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
        
        #Like StackHorizontal, but with padding
        def merge_horizontal(left, right):
            lwidth = left.width - pad * self.model_scale
            rwidth = right.width - pad * self.model_scale
            if (left.height, left.num_frames, left.format) != (right.height, right.num_frames, right.format):
                raise Exception("Left and right clip must have the same height, format and number of frames!")
            mask = make_gradient(pad * 2 * self.model_scale, left.height, left.num_frames)
            mask = core.std.AddBorders(mask, left = lwidth - pad * self.model_scale, color = [0.0])
            mask = core.std.AddBorders(mask, right = rwidth - pad * self.model_scale, color = [1.0])
            left = core.std.AddBorders(left, right = rwidth - pad * self.model_scale)
            right = core.std.AddBorders(right, left = lwidth - pad * self.model_scale)
            return core.std.MaskedMerge(left, right, mask)

        # Like StackVertical, but with padding
        def merge_vertical(top, bottom):
            theight = top.height - pad * self.model_scale
            bheight = bottom.height - pad * self.model_scale
            if (top.width, top.num_frames, top.format) != (bottom.width, bottom.num_frames, bottom.format):
                raise Exception("Top and bottom clip must have the same height, format and number of frames!")
            mask = make_gradient(top.width, pad * 2 * self.model_scale, top.num_frames, vertical = True)
            mask = core.std.AddBorders(mask, top = theight - pad * self.model_scale, color = [0.0])
            mask = core.std.AddBorders(mask, bottom = bheight - pad * self.model_scale, color = [1.0])
            top = core.std.AddBorders(top, bottom = bheight - pad * self.model_scale)
            bottom = core.std.AddBorders(bottom, top = theight - pad * self.model_scale)
            return core.std.MaskedMerge(top, bottom, mask)

        # Remember the clip's original format
        original_format = clip.format
        if (original_format.color_family, original_format.bits_per_sample, original_format.sample_type) != (vs.RGB, 32, vs.FLOAT):
            raise Exception("VSGAN input must be 32 bit RGB, not " + original_format.name)
        buffer = clip
        # Convert clip to floating point RGB, as that's what ESRGAN uses internally
        #import mvsfunc as mvf
        #buffer = core.resize.Spline36(clip, format = vs.RGBS)
        #buffer = mvf.ToRGB(clip, depth = 32)
        # Send the clip array to execute()
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
        # If chunked, rejoin the chunked clips, otherwise return the result.
        if chunk:
            tophalf = merge_horizontal(results[0], results[1])
            bottomhalf = merge_horizontal(results[2], results[3])
            buffer = merge_vertical(tophalf, bottomhalf)
        else:
            buffer = results[0]
        # Optionally convert back to the original color space
        # Note that (unlike ToRGB), mvs.ToYUV() guesses the original color space
        # based on whether the RGB clip is SD, HD, or UHD.
        # Upscaling is going to influence this decision, 
        # so it's usually best to leave the output unchanged
        # and let the user reconvert it with the appropriate arguments. 
        if reconvert and (original_format.color_family != buffer.format.color_family):
            if original_format.color_family == vs.ColorFamily.RGB:
                buffer = core.resize.Spline36(buffer, format = original_format)
            if original_format.color_family == vs.ColorFamily.YUV:
                buffer = core.resize.Spline36(buffer, format = original_format, matrix_s = matrix_s)
                #import mvsfunc
                #buffer = mvsfunc.ToYUV(buffer, css=original_format.name[3:6])
                
        # return the new frame
        return buffer

    def convert_new_to_old(state_dict):
        old_net = {}
        items = []
        for k, v in state_dict.items():
            items.append(k)

        old_net['model.0.weight'] = state_dict['conv_first.weight']
        old_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                old_net[ori_k] = state_dict[k]
                items.remove(k)

        old_net['model.1.sub.23.weight'] = state_dict['trunk_conv.weight']
        old_net['model.1.sub.23.bias'] = state_dict['trunk_conv.bias']
        old_net['model.3.weight'] = state_dict['upconv1.weight']
        old_net['model.3.bias'] = state_dict['upconv1.bias']
        old_net['model.6.weight'] = state_dict['upconv2.weight']
        old_net['model.6.bias'] = state_dict['upconv2.bias']
        old_net['model.8.weight'] = state_dict['HRconv.weight']
        old_net['model.8.bias'] = state_dict['HRconv.bias']
        old_net['model.10.weight'] = state_dict['conv_last.weight']
        old_net['model.10.bias'] = state_dict['conv_last.bias']
        return old_net
    
    def get_rrdb_net_arch(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        """
        Import RRDB Net Architecture
        """
         # in_nc = 3  # num of input channels
        # out_nc = 3  # num of output channels
        # nf = 64  # num of filters
        # nb = 23  # num of blocks todo
        # gc = 32

        from . import RRDBNet_arch_old as Arch
        return Arch.RRDB_Net(
            in_nc, out_nc, nf, nb, gc,
            upscale=self.model_scale,
            norm_type=None,
            act_type="leakyrelu",
            mode="CNA",
            res_scale=1,
            upsample_mode="upconv"
        )

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
    def cv2_imwrite(image, out_color_space="RGBS"): #TODO: "RGBS"
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
                #d = 
                # copy the value of d, into s[frame_num, :, :, plane_num]
                np.copyto(np.array(frame.get_write_array(plane_num), copy=False), image[n, :, :, i], casting="unsafe")
                # delete the d variable from memory
                #del d
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
        #img = img * 1.0 / 255 #RGB24 -> RGBS
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_lr = img.unsqueeze(0)
        img_lr = img_lr.to(self.torch_device)
        with torch.no_grad():
            output = self.rrdb_net_model(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        #output = (output * 255.0).round() #RGBS -> RGB24
        return self.cv2_imwrite(image=output, out_color_space=clip.format.name)
