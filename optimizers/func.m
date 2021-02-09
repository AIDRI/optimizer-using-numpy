tx = -4:0.2:4;
ty = -4:0.2:4;

[x, y] = meshgrid (tx, ty);
tz = 3.*(1.-x).^2.*exp(-x.^2 .- (y.+1).^2) .- 10.*(x./5 .- x.^3 .- y.^5).*exp(-x.^2.-y.^2 + 1.5) .- 1/3.*exp(-(x.+1).^2 .- y.^2);
surf (tx, ty, tz);
shading interp;