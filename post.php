<?php

$data = '{"filename":"test_name"';
$data .= ',"filetype":"test_typ"}';

$ch = curl_init(); 
curl_setopt($ch, CURLOPT_URL, "http://localhost:5000/api?url_post=post_url"); 
curl_setopt($ch, CURLOPT_POST, 1);
curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1); 
curl_setopt($ch, CURLOPT_HTTPHEADER, array(
	'Content-Type: application/json'
));
$output = curl_exec($ch); 
curl_close($ch);  

echo $output;

?>
